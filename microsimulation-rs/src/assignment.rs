use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use csv::Writer;
use hashbrown::HashSet;
use log::{error, info, warn};
use polars::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use typed_index_collections::TiVec;

use crate::{
    config::{Config, Year},
    person::ChildHRPerson,
    queues::AdultOrChild,
    ADULT_AGE, OA,
};
use crate::{digest, Eth};
use crate::{
    household::{Household, HID},
    queues::Queues,
};
use crate::{
    person::{HRPerson, PartnerHRPerson, Person, HRPID, PID},
    MSOA,
};

// TODO: remove, temporary helper for debugging.
fn _input() {
    std::io::stdin().read_exact(&mut [0]).unwrap();
}

#[derive(Debug)]
pub struct Assignment {
    pub region: String,
    pub year: Year,
    pub output_dir: PathBuf,
    pub scotland: bool,
    pub h_data: TiVec<HID, Household>,
    pub p_data: TiVec<PID, Person>,
    pub strictmode: bool,
    pub geog_lookup: DataFrame,
    pub hrp_dist: BTreeMap<String, TiVec<HRPID, HRPerson>>,
    pub hrp_index: BTreeMap<String, Vec<i32>>,
    pub partner_hrp_dist: TiVec<HRPID, PartnerHRPerson>,
    pub child_hrp_dist: TiVec<HRPID, ChildHRPerson>,
    pub queues: Queues,
    pub rng: StdRng,
}

fn read_geog_lookup(path: impl Into<PathBuf>) -> anyhow::Result<DataFrame> {
    let mut df = CsvReader::from_path(path)?.finish()?;
    df.rename("OA", "oa")?
        .rename("MSOA", "msoa")?
        .rename("LAD", "la")?
        .rename("LSOA", "lsoa")?;
    Ok(df)
}

// See example: https://docs.rs/polars/latest/polars/frame/struct.DataFrame.html#method.apply
fn _replace_i32(mapping: &HashMap<i32, i32>) -> impl (Fn(&Series) -> Series) + '_ {
    |series: &Series| -> Series {
        series
            .cast(&DataType::Int32)
            .unwrap()
            .i32()
            .unwrap()
            .into_iter()
            .map(|opt_el: Option<i32>| opt_el.map(|el| *mapping.get(&el).unwrap_or(&el)))
            .collect::<Int32Chunked>()
            .into_series()
    }
}

pub fn read_csv<P: AsRef<Path>, K, V: for<'a> Deserialize<'a>>(
    path: P,
) -> anyhow::Result<TiVec<K, V>> {
    Ok(csv::Reader::from_path(path)?
        .deserialize()
        .collect::<Result<TiVec<K, V>, _>>()?)
}

trait GetSetEth {
    fn get_eth(&self) -> &Eth;
    fn set_eth(&mut self, eth: Eth);
}

impl GetSetEth for Person {
    fn get_eth(&self) -> &Eth {
        &self.eth
    }
    fn set_eth(&mut self, eth: Eth) {
        self.eth = eth;
    }
}

impl GetSetEth for Household {
    fn get_eth(&self) -> &Eth {
        &self.lc4202_c_ethhuk11
    }

    fn set_eth(&mut self, eth: Eth) {
        self.lc4202_c_ethhuk11 = eth;
    }
}

fn map_eth<K, V: GetSetEth>(
    data: TiVec<K, V>,
    eth_mapping: &HashMap<Eth, Eth>,
) -> anyhow::Result<TiVec<K, V>> {
    data.into_iter()
        .map(|mut person| {
            match eth_mapping
                // TODO: fix int types
                .get(person.get_eth())
                .cloned()
                .ok_or(anyhow!("Invalid mapping."))
            {
                Ok(new_val) => {
                    person.set_eth(new_val);
                    Ok(person)
                }
                Err(e) => Err(e),
            }
        })
        .collect::<anyhow::Result<TiVec<K, V>>>()
}

enum Parent {
    Single,
    Couple,
}

impl Assignment {
    pub fn new(region: &str, rng_seed: u64, config: &Config) -> anyhow::Result<Assignment> {
        let h_file = config.data_dir.join(format!(
            "ssm_hh_{}_{}_{}.csv",
            region, config.household_resolution, config.year
        ));
        let p_file = config.data_dir.join(format!(
            "ssm_{}_{}_{}_{}.csv",
            region, config.person_resolution, config.projection, config.year
        ));

        let geog_lookup = read_geog_lookup(config.persistent_data().join("gb_geog_lookup.csv.gz"))?;
        let mut hrp_dist: BTreeMap<String, TiVec<HRPID, HRPerson>> = BTreeMap::new();
        hrp_dist.insert(
            "sgl".to_string(),
            read_csv(config.persistent_data().join("hrp_sgl_dist.csv"))?,
        );
        hrp_dist.insert(
            "cpl".to_string(),
            read_csv(config.persistent_data().join("hrp_cpl_dist.csv"))?,
        );
        hrp_dist.insert(
            "sp".to_string(),
            read_csv(config.persistent_data().join("hrp_sp_dist.csv"))?,
        );
        hrp_dist.insert(
            "mix".to_string(),
            read_csv(config.persistent_data().join("hrp_dist.csv"))?,
        );

        let mut hrp_index: BTreeMap<String, Vec<i32>> = BTreeMap::new();
        hrp_index.insert("sgl".to_string(), vec![1]);
        hrp_index.insert("cpl".to_string(), vec![2, 3]);
        hrp_index.insert("sp".to_string(), vec![4]);
        hrp_index.insert("mix".to_string(), vec![5]);

        let partner_hrp_dist = read_csv(config.persistent_data().join("partner_hrp_dist.csv"))?;
        let child_hrp_dist = read_csv(config.persistent_data().join("child_hrp_dist.csv"))?;
        let scotland = region.starts_with('S');
        let mut h_data: TiVec<HID, Household> = read_csv(h_file)?;
        let mut p_data: TiVec<PID, Person> = read_csv(p_file)?;

        // TODO: check the mapping
        if !scotland {
            let eth_mapping = [
                (-1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 4),
                (7, 5),
                (8, 5),
                (9, 5),
                (10, 5),
                (12, 6),
                (13, 6),
                (14, 6),
                (15, 6),
                (16, 6),
                (18, 7),
                (19, 7),
                (20, 7),
                (22, 8),
                (23, 8),
            ]
            .into_iter()
            .map(|(x, y)| (Eth(x), Eth(y)))
            .collect::<HashMap<Eth, Eth>>();
            p_data = map_eth(p_data, &eth_mapping)?;
        } else {
            // TODO: check the mapping
            let eth_mapping = [(-1, 1), (1, 1), (8, 2), (9, 3), (15, 4), (18, 5), (22, 6)]
                .into_iter()
                .map(|(x, y)| (Eth(x), Eth(y)))
                .collect::<HashMap<Eth, Eth>>();
            let eth_remapping = [(-1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8)]
                .into_iter()
                .map(|(x, y)| (Eth(x), Eth(y)))
                .collect::<HashMap<Eth, Eth>>();
            p_data = map_eth(p_data, &eth_mapping)?;
            p_data = map_eth(p_data, &eth_remapping)?;
            // TODO: check should there be a h_data mapping call here first?
            h_data = map_eth(h_data, &eth_remapping)?;
        }

        // Assert eth mapping performed correctly
        p_data
            .iter()
            .for_each(|person| assert!((1..=8).contains(&person.eth.0)));

        let mut rng = StdRng::seed_from_u64(rng_seed);
        let queues = Queues::new(&p_data, &mut rng);
        Ok(Self {
            region: region.to_owned(),
            year: config.year.to_owned(),
            output_dir: config.data_dir.to_owned(),
            scotland,
            h_data,
            p_data,
            strictmode: config.strict,
            geog_lookup,
            hrp_dist,
            hrp_index,
            partner_hrp_dist,
            child_hrp_dist,
            queues,
            rng,
        })
    }

    /// Generate digest for people and households.
    pub fn digest(&self) -> anyhow::Result<String> {
        digest((&self.p_data, &self.h_data))
    }

    fn sample_hrp(&mut self, msoa: &MSOA, oas: &HashSet<OA>) -> anyhow::Result<()> {
        for hh_type in ["sgl", "cpl", "sp", "mix"]
            .into_iter()
            .map(|s| s.to_owned())
        {
            let hrp_dist = self.hrp_dist.get(&hh_type).unwrap();
            let weighted_idx = WeightedIndex::new(hrp_dist.iter().map(|hrp| hrp.n)).unwrap();
            let idxs = self.hrp_index.get(&hh_type).unwrap();
            let h_ref: Vec<_> = self
                .h_data
                .iter_mut()
                .filter(|household| {
                    idxs.contains(&household.lc4408_c_ahthuk11)
                        && oas.contains(&household.oa)
                        && household.hrpid.is_none()
                })
                .collect();

            if h_ref.is_empty() {
                continue;
            }

            // Get sample of HRPs
            let sample = (0..h_ref.len())
                .map(|_| {
                    let hrpid = HRPID(weighted_idx.sample(&mut self.rng));
                    hrp_dist.get(hrpid).unwrap()
                })
                .collect::<Vec<_>>();

            // Loop over sample HRPs and match PIDs
            for (sample_person, household) in sample.iter().zip(h_ref) {
                // Demographics
                let age = sample_person.age;
                let sex = sample_person.sex;
                let eth = sample_person.eth;

                // Note: Currently possible for the HRP to be a child (as in python, should it be
                // impossible?), but different sampling is implemented in python where
                // `get_closest_adult()` is called for all cases:
                // https://github.com/alan-turing-institute/microsimulation/blob/2373691bd0ff764db129e52ec78d71c58538d9af/microsimulation/assignment.py#L226
                // TODO: consider fix in `sample_person` below to match python version.
                // It will have negligible effect as rare event that HRP is child.
                let adult_or_child = if age > ADULT_AGE {
                    AdultOrChild::Adult
                } else {
                    warn!("HRP is child");
                    AdultOrChild::Child
                };

                // Try exact match over unmatched
                if let Some(pid) =
                    self.queues
                        .sample_person(msoa, age, sex, eth, adult_or_child, &self.p_data)
                {
                    // Assign pid to household
                    household.hrpid = Some(pid);
                    // Assign household to person
                    self.p_data
                        .get_mut(pid)
                        .unwrap_or_else(|| panic!("Invalid {pid}"))
                        .hid = Some(household.hid);

                    // If single person household, filled
                    if household.lc4408_c_ahthuk11 == 1 {
                        household.filled = Some(true)
                    }
                    self.queues.debug_stats(pid);
                } else {
                    return Err(anyhow!("No match for: {sample_person:?}").context("sample_hrp()"));
                }
            }
        }
        Ok(())
    }

    fn sample_partner(&mut self, msoa: &MSOA, oas: &HashSet<OA>) -> anyhow::Result<()> {
        let h2_ref: Vec<_> = self
            .h_data
            .iter_mut()
            .filter(|household| {
                [2, 3].contains(&household.lc4408_c_ahthuk11)
                    && oas.contains(&household.oa)
                    && household.filled != Some(true)
            })
            .collect();

        // Sampling by age and ethnicity
        let mut dist_by_ae = HashMap::new();
        // Sampling by age
        let mut dist_by_a = HashMap::new();
        // Populate lookups
        self.partner_hrp_dist
            .iter_enumerated()
            .for_each(|(hrpid, partner)| {
                dist_by_ae
                    .entry((partner.agehrp, partner.eth))
                    .and_modify(|v: &mut Vec<(HRPID, usize)>| {
                        v.push((hrpid, partner.n));
                    })
                    .or_insert(vec![(hrpid, partner.n)]);
                dist_by_a
                    .entry(partner.agehrp)
                    .and_modify(|v: &mut Vec<(HRPID, usize)>| {
                        v.push((hrpid, partner.n));
                    })
                    .or_insert(vec![(hrpid, partner.n)]);
            });
        // Construct vec off HRPID and weights
        let dist = self
            .partner_hrp_dist
            .iter_enumerated()
            .map(|(idx, person)| (idx.to_owned(), person.n))
            .collect();
        for household in h2_ref {
            let hrpid = household.hrpid.expect("Household is not assigned a PID");
            let hrp = self.p_data.get(hrpid).expect("Invalid HRPID");
            let hrp_sex = hrp.sex;
            let hrp_age = hrp.age;
            let hrp_eth = hrp.eth;

            // Pick dist
            let dist = dist_by_ae.get(&(hrp_age, hrp_eth)).unwrap_or_else(|| {
                warn!(
                    "Partner-HRP not sampled: {}, {}, {} - resample withouth eth",
                    hrp_age, hrp_sex, hrp_eth
                );
                dist_by_a.get(&hrp_age).unwrap_or_else(|| {
                    warn!(
                        "Partner-HRP not sampled: {}, {}, {}",
                        hrp_age, hrp_sex, hrp_eth
                    );
                    &dist
                })
            });

            let partner_sample_id = dist.choose_weighted(&mut self.rng, |item| item.1)?.0;
            let partner_sample = self
                .partner_hrp_dist
                .get(partner_sample_id)
                .ok_or(anyhow!("Invalid HRPID: {partner_sample_id}"))?;
            let age = partner_sample.age;
            let sex = if partner_sample.samesex {
                hrp_sex
            } else {
                hrp_sex.opposite()
            };

            // TODO: check why this is `.ethnicityew` and not `.eth`
            let eth: Eth = partner_sample.ethnicityew.into();

            // Sample a HR person
            if let Some(pid) =
                self.queues
                    .sample_person(msoa, age, sex, eth, AdultOrChild::Adult, &self.p_data)
            {
                // Assign household to person
                self.p_data
                    .get_mut(pid)
                    .unwrap_or_else(|| panic!("Invalid {pid}"))
                    .hid = Some(household.hid);

                // If single person household, filled
                if household.lc4404_c_sizhuk11 == 2 {
                    household.filled = Some(true)
                }
                self.queues.debug_stats(pid);
            } else {
                // TODO consider returning error variant instead of logging
                error!("No partner match!");
            }
        }

        Ok(())
    }

    fn sample_child(
        &mut self,
        msoa: &MSOA,
        oas: &HashSet<OA>,
        num_occ: i32,
        mark_filled: bool,
        parent: Parent,
    ) -> anyhow::Result<()> {
        let hsp_ref: Vec<_> = self
            .h_data
            .iter_mut()
            .filter(|household| {
                // TODO: check condition as for other sample
                oas.contains(&household.oa)
                    && household.lc4404_c_sizhuk11.eq(&num_occ)
                    && match parent {
                        Parent::Single => household.lc4408_c_ahthuk11.eq(&4),
                        Parent::Couple => [2, 3].contains(&household.lc4408_c_ahthuk11),
                    }
                    && household.filled != Some(true)
            })
            .collect();

        // Sampling by age and ethnicity
        let mut dist_by_ae = HashMap::new();
        // Sampling by age
        let mut dist_by_a = HashMap::new();
        // Sampling by eth
        let mut dist_by_e = HashMap::new();

        // Populate lookups
        self.child_hrp_dist
            .iter_enumerated()
            // TODO: check the conditions with the dataframe filter
            .for_each(|(hrpid, child)| {
                dist_by_ae
                    .entry((child.agehrp, child.eth))
                    .and_modify(|v: &mut Vec<(HRPID, usize)>| {
                        v.push((hrpid, child.n));
                    })
                    .or_insert(vec![(hrpid, child.n)]);
                dist_by_a
                    .entry(child.agehrp)
                    .and_modify(|v: &mut Vec<(HRPID, usize)>| {
                        v.push((hrpid, child.n));
                    })
                    .or_insert(vec![(hrpid, child.n)]);
                dist_by_e
                    .entry(child.eth)
                    .and_modify(|v: &mut Vec<(HRPID, usize)>| {
                        v.push((hrpid, child.n));
                    })
                    .or_insert(vec![(hrpid, child.n)]);
            });

        // Construct vec from HRPID and weights
        let dist = self
            .child_hrp_dist
            .iter_enumerated()
            .map(|(idx, person)| (idx.to_owned(), person.n))
            .collect();

        for household in hsp_ref {
            let hrpid = household.hrpid.expect("HRPID is not assigned.");
            let hrp_person = self.p_data.get(hrpid).expect("Invalid PID.");
            let hrp_age = hrp_person.age;
            // TODO: check why unused
            let hrp_sex = hrp_person.sex;
            let hrp_eth = hrp_person.eth;

            // Pick dist
            let dist = match parent {
                Parent::Single => dist_by_ae.get(&(hrp_age, hrp_eth)).unwrap_or_else(|| {
                    dist_by_a
                        .get(&hrp_age)
                        .unwrap_or_else(|| dist_by_e.get(&hrp_eth).unwrap_or(&dist))
                }),
                Parent::Couple => {
                    if let Some(dist) = dist_by_ae.get(&(hrp_age, hrp_eth))
                    // TODO: confirm handling: assignment.py, L437-L440
                    // .unwrap_or_else(|| panic!("No matching {hrp_age}, {hrp_eth} in distribution.")),
                    {
                        dist
                    } else {
                        warn!(
                            "child-HRP not sampled: {}, {}, {}",
                            hrp_age, hrp_sex, hrp_eth
                        );
                        continue;
                    }
                }
            };
            // Sample:: TODO: make sep fn
            let child_sample_id = dist.choose_weighted(&mut self.rng, |item| item.1)?.0;
            let child_sample = self
                .child_hrp_dist
                .get(child_sample_id)
                .ok_or(anyhow!("Invalid HRPID: {child_sample_id}"))?;
            let age = child_sample.age;
            let sex = child_sample.sex;

            // TODO: check handling see L392 of assignment.py:
            // https://github.com/alan-turing-institute/microsimulation/blob/2373691bd0ff764db129e52ec78d71c58538d9af/microsimulation/assignment.py#L392
            // i.e. why is this `.ethnicityew` and not `.eth`
            let eth: Eth = child_sample.ethnicityew.into();

            // Get match from population
            if let Some(pid) =
                self.queues
                    .sample_person(msoa, age, sex, eth, AdultOrChild::Child, &self.p_data)
            {
                self.p_data
                    .get_mut(pid)
                    .unwrap_or_else(|| panic!("Invalid {pid}"))
                    .hid = Some(household.hid);
                if mark_filled {
                    household.filled = Some(true)
                }
                self.queues.debug_stats(pid);
            } else {
                warn!(
                    "child not found,  age: {}, sex: {:?}, eth: {:?}",
                    age, sex, eth
                );
            }
        }

        Ok(())
    }

    fn fill_multi(
        &mut self,
        msoa: &MSOA,
        oas: &HashSet<OA>,
        nocc: i32,
        mark_filled: bool,
    ) -> anyhow::Result<()> {
        let mut h_ref: Vec<_> = self
            .h_data
            .iter_mut()
            .filter(|household| {
                // TODO: check condition as for other sample
                oas.contains(&household.oa)
                    && household.lc4408_c_ahthuk11.eq(&5)
                    && household.filled != Some(true)
            })
            .collect();

        for (idx, household) in h_ref.iter_mut().enumerate() {
            let pid = self.queues.sample_adult_any(msoa);
            // TODO: create method for assignment part
            if let Some(pid) = pid {
                self.p_data
                    .get_mut(pid)
                    .unwrap_or_else(|| panic!("Invalid {pid}"))
                    .hid = Some(household.hid);
                // Mark households as filled if size is equal to given arguments
                if mark_filled && household.lc4404_c_sizhuk11.eq(&nocc) {
                    household.filled = Some(true);
                }
                self.queues.debug_stats(pid);
            } else {
                warn!(
                    "Out of multi-people, need {} households for {}",
                    h_ref.len(),
                    idx + 1
                );
                break;
            }
        }
        Ok(())
    }

    fn fill_communal(&mut self, msoa: &MSOA, oas: &HashSet<OA>) -> anyhow::Result<()> {
        let mut c_ref: Vec<_> = self
            .h_data
            .iter_mut()
            .filter(|household| {
                oas.contains(&household.oa) && household.qs420_cell > -1
                // TODO: check this condition
                // && household.filled != Some(true)
            })
            .collect();

        for household in c_ref.iter_mut() {
            let ctype = household.qs420_cell;

            let nocc = household.communal_size;
            if nocc > 0 {
                // TODO: refactor into method on queues
                // Get samples of nocc size
                let mut pids = vec![];
                while i32::try_from(pids.len()).expect("Not i32") < nocc {
                    let pid = if ctype < 22 {
                        self.queues.sample_person_over_75(msoa)
                    } else if ctype < 27 {
                        self.queues.sample_person_19_to_25(msoa)
                    } else {
                        self.queues.sample_person_over_16(msoa)
                    };
                    if let Some(pid) = pid {
                        pids.push(pid);
                    } else {
                        break;
                    }
                }
                if i32::try_from(pids.len()).expect("Not i32").lt(&nocc) {
                    warn!("cannot assign to communal: {:?}", household);
                    // Put PIDs back
                    // TODO: refactor into method on queues
                    while let Some(pid) = pids.pop() {
                        let map = if ctype < 22 {
                            &mut self.queues.people_by_area_over_75
                        } else if ctype < 27 {
                            &mut self.queues.people_by_area_19_to_25
                        } else {
                            &mut self.queues.people_by_area_over_16
                        };
                        map.get_mut(msoa)
                            .expect("MSOA does not exist in lookup")
                            .push(pid);

                        // Remove PID from matched set
                        self.queues.matched.remove(&pid);
                    }
                    // Continue as could not fill
                    continue;
                }

                while let Some(pid) = pids.pop() {
                    self.p_data
                        .get_mut(pid)
                        .unwrap_or_else(|| panic!("Invalid {pid}"))
                        .hid = Some(household.hid);
                    self.queues.debug_stats(pid);
                }
            }
            household.filled = Some(true);
        }
        Ok(())
    }
    fn assign_surplus_adults(&mut self, msoa: &MSOA, oas: &HashSet<OA>) -> anyhow::Result<()> {
        let p_unassigned: Vec<&mut Person> = self
            .p_data
            .iter_mut()
            .filter_map(|person| {
                if person.msoa.eq(msoa) && person.age > ADULT_AGE && person.hid.is_none() {
                    Some(person)
                } else {
                    None
                }
            })
            .collect();

        let h_candidates: Vec<_> = self
            .h_data
            .iter_mut()
            .filter(|household| {
                oas.contains(&household.oa)
                    && household.lc4408_c_ahthuk11.eq(&5)
                    && household.filled != Some(true)
            })
            .collect();
        if !h_candidates.is_empty() {
            for person in p_unassigned {
                let h_sample = h_candidates
                    .choose(&mut self.rng)
                    .expect("Cannot be empty.");
                person.hid = Some(h_sample.hid);
                let pid = person.pid;
                self.queues.matched.insert(pid);
                self.queues.debug_stats(pid);
            }
        }
        Ok(())
    }

    fn assign_surplus_children(&mut self, msoa: &MSOA, oas: &HashSet<OA>) -> anyhow::Result<()> {
        for eth in [2, 3, 4, 5, 6, 7, 8].into_iter().map(Eth) {
            let c_unassigned: Vec<&mut Person> = self
                .p_data
                .iter_mut()
                .filter_map(|person| {
                    if person.msoa.eq(msoa)
                        && person.age <= ADULT_AGE
                        && person.hid.is_none()
                        && person.eth.eq(&eth)
                    {
                        Some(person)
                    } else {
                        None
                    }
                })
                .collect();

            let h_candidates: Vec<_> = self
                .h_data
                .iter_mut()
                .filter(|household| {
                    oas.contains(&household.oa)
                        && household.lc4202_c_ethhuk11.eq(&eth)
                        && [2, 3, 4, 5].contains(&household.lc4408_c_ahthuk11)
                        && household.filled != Some(true)
                })
                .collect();
            if !h_candidates.is_empty() && c_unassigned.len().gt(&0) {
                for person in c_unassigned {
                    let h_sample = h_candidates
                        .choose(&mut self.rng)
                        .expect("Cannot be empty.");
                    person.hid = Some(h_sample.hid);
                    let pid = person.pid;
                    self.queues.matched.insert(pid);
                }
            }
        }
        Ok(())
    }

    pub fn info_stats(&self) {
        let assigned_people = self
            .p_data
            .iter()
            .filter(|person| person.hid.is_some())
            .count();
        let assigned_households = self
            .h_data
            .iter()
            .filter(|household| household.filled.eq(&Some(true)))
            .count() as f64;
        let total_people = self.p_data.len() as f64;
        let total_households = self
            .h_data
            .iter()
            .filter(|household| household.lc4408_c_ahthuk11 > 0)
            .count() as f64;
        info!(
            "{0:25}: {1:6} ({2:3.2}%)",
            "People",
            assigned_people,
            100. * (assigned_people as f64 / total_people)
        );
        info!(
            "{0:25}: {1:6}",
            "Remaining people",
            self.p_data
                .iter()
                .filter(|person| person.hid.is_none())
                .count()
        );
        info!(
            "{0:25}: {1:6} ({2:3.2})%",
            "Households",
            assigned_households,
            100. * assigned_households / total_households
        );
        info!(
            "{0:25}: {1:6} (+{2:6})",
            "Remaining households",
            self.h_data
                .iter()
                .filter(|household| !household.filled.eq(&Some(true))
                    && household.lc4408_c_ahthuk11 > 0)
                .count(),
            self.h_data
                .iter()
                .filter(|household| household.lc4408_c_ahthuk11.eq(&-1))
                .count()
        )
    }

    pub fn check(&self) -> anyhow::Result<()> {
        info!("---");
        info!("Checking...");
        info!("---");
        info!(
            "Occupied households without HRP: {}",
            self.h_data
                .iter()
                .filter(|household| household.lc4408_c_ahthuk11 > 0 && household.hrpid.is_none())
                .count()
        );
        info!(
            "Occupied households not filled: {} of {}",
            self.h_data
                .iter()
                .filter(|household| household.lc4408_c_ahthuk11 > 0 && household.filled.is_none())
                .count(),
            self.h_data
                .iter()
                .filter(|household| household.lc4408_c_ahthuk11 > 0)
                .count()
        );
        info!(
            "Communal residences not filled: {}",
            self.h_data
                .iter()
                .filter(|household| household.communal_size.ge(&0) && household.filled.is_none())
                .count()
        );
        info!("Single-occupant households not filled: {}",
            self.h_data
                .iter()
                .filter(|household| household.lc4408_c_ahthuk11.eq(&1) && household.filled.is_none())
                .count(),

        );
        info!(
            "Single-parent one-child households not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    household.lc4408_c_ahthuk11.eq(&4)
                        && household.lc4404_c_sizhuk11.eq(&2)
                        && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Single-parent two-child households not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    household.lc4408_c_ahthuk11.eq(&4)
                        && household.lc4404_c_sizhuk11.eq(&3)
                        && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Single-parent 3+ households not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    household.lc4408_c_ahthuk11.eq(&4)
                        && household.lc4404_c_sizhuk11.eq(&4)
                        && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Couple households with no children not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    [2, 3].contains(&household.lc4408_c_ahthuk11)
                        && household.lc4404_c_sizhuk11.eq(&2)
                        && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Couple households with one child not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    [2, 3].contains(&household.lc4408_c_ahthuk11)
                        && household.lc4404_c_sizhuk11.eq(&3)
                        && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Couple households with 2+ children not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    [2, 3].contains(&household.lc4408_c_ahthuk11)
                    // TODO: should this be .ge(&4) if it is 2+ (not in python)
                    // && household.lc4404_c_sizhuk11.eq(&4)
                    && household.lc4404_c_sizhuk11.ge(&4)
                    && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Mixed (2,3) households not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    household.lc4408_c_ahthuk11.eq(&5)
                        && household.lc4404_c_sizhuk11.lt(&4)
                        && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Mixed (4+) households not filled: {}",
            self.h_data
                .iter()
                .filter(|household| {
                    household.lc4408_c_ahthuk11.eq(&5)
                    // TODO: should the next line be included here (not in python) this be given 4+
                    && household.lc4404_c_sizhuk11.ge(&4)
                    && household.filled.is_none()
                })
                .count(),
        );
        info!(
            "Adults not assigned {} of {}",
            self.p_data
                .iter()
                .filter(|person| { person.age.gt(&ADULT_AGE) && person.hid.is_none() })
                .count(),
            self.p_data
                .iter()
                .filter(|person| { person.age.gt(&ADULT_AGE) })
                .count()
        );
        info!(
            "Children not assigned {} of {}",
            self.p_data
                .iter()
                .filter(|person| { person.age.le(&ADULT_AGE) && person.hid.is_none() })
                .count(),
            self.p_data
                .iter()
                .filter(|person| { person.age.le(&ADULT_AGE) })
                .count()
        );
        Ok(())
    }

    // TODO: add type for LAD
    pub fn run(&mut self) -> anyhow::Result<()> {
        // Deterministic ordering
        let msoas: BTreeSet<MSOA> = self
            .p_data
            .iter()
            .map(|person| person.msoa.to_owned())
            .collect();

        // Run assignment over each MSOA
        // TODO: add random permutation over MSOAs
        for msoa in msoas.iter() {
            let oas = self
                .geog_lookup
                .clone()
                .lazy()
                .filter(col("msoa").eq(lit(String::from(msoa.to_owned()))))
                .select([col("oa")])
                .collect()?;
            let oas: HashSet<OA> = oas
                .iter()
                .next()
                .unwrap()
                .str()?
                .into_iter()
                .map(|el| el.unwrap().to_owned().into())
                .collect();
            info!(">>> MSOA: {}", msoa);
            info!(
                ">>> OAs : {}",
                oas.iter()
                    .map(|oa| oa.0.to_owned())
                    .collect::<Vec<String>>()
                    .join(", ")
            );

            // Sample HRP
            info!(">>> Assigning HRPs");
            self.sample_hrp(msoa, &oas)?;
            self.info_stats();

            // TODO: mark_filled bools set to match python, check these are correct

            // Sample partner
            // TODO: check all partners assigned (from python)
            info!(">>> Assigning partners to HRPs where appropriate");
            self.sample_partner(msoa, &oas)?;
            self.info_stats();

            info!(">>> Assigning child 1 to single-parent households");
            self.sample_child(msoa, &oas, 2, true, Parent::Single)?;
            self.info_stats();

            info!(">>> Assigning child 2 to single-parent households");
            self.sample_child(msoa, &oas, 3, true, Parent::Single)?;
            self.info_stats();

            info!(">>> Assigning child 3 to single-parent households");
            self.sample_child(msoa, &oas, 4, false, Parent::Single)?;
            self.info_stats();

            // # TODO if partner hasnt been assigned then household may be incorrectly marked filled
            info!(">>> Assigning child 1 to couple households");
            self.sample_child(msoa, &oas, 3, true, Parent::Couple)?;
            self.info_stats();

            // # TODO if partner hasnt been assigned then household may be incorrectly marked filled
            info!(">>> Assigning child 2 to couple households");
            self.sample_child(msoa, &oas, 4, false, Parent::Couple)?;
            self.info_stats();

            info!(">>> Multi-person households");
            self.fill_multi(msoa, &oas, 2, true)?;
            self.fill_multi(msoa, &oas, 3, true)?;
            self.fill_multi(msoa, &oas, 4, false)?;
            self.info_stats();

            info!(">>> Assigning people to communal establishments");
            self.fill_communal(msoa, &oas)?;
            self.info_stats();

            info!(">>> Assigning surplus adults");
            self.assign_surplus_adults(msoa, &oas)?;
            self.info_stats();

            info!(">>> Assigning surplus children");
            self.assign_surplus_children(msoa, &oas)?;
            self.info_stats();
        }

        Ok(())
    }

    // TODO: implement write record
    pub fn write(&self, region: &str, config: &Config) -> anyhow::Result<()> {
        let dir = "outputs/";
        std::fs::create_dir_all(dir)?;

        // Serialize people
        // TODO: wrap in function
        let mut writer = Writer::from_writer(vec![]);
        self.p_data.iter().for_each(|person| {
            writer.serialize(person).unwrap();
        });
        let data = String::from_utf8(writer.into_inner()?)?;
        let path = format!(
            "{dir}/rs_ass_{}_{}_{}.csv",
            region, config.person_resolution, config.year
        );
        std::fs::write(path, data)?;

        // Serialize households
        let mut writer = Writer::from_writer(vec![]);
        self.h_data.iter().for_each(|household| {
            writer.serialize(household).unwrap();
        });
        let data = String::from_utf8(writer.into_inner()?)?;
        let path = format!(
            "{dir}/rs_ass_hh_{}_{}_{}.csv",
            region, config.household_resolution, config.year
        );
        std::fs::write(path, data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use lazy_static::lazy_static;

    use crate::config::{Projection, Resolution};

    use super::*;

    const PERSISTENT_DATA_DIR: &str = "../persistent_data/";

    lazy_static! {
        static ref TEST_CONFIG: Config = Config {
            person_resolution: Resolution::MSOA11,
            household_resolution: Resolution::OA11,
            projection: Projection::PPP,
            strict: false,
            year: Year(2020),
            data_dir: PathBuf::from_str("tests/data/").unwrap(),
            persistent_data_dir: Some(PathBuf::from_str(PERSISTENT_DATA_DIR).unwrap()),
            profile: false,
        };
        static ref ENV_LOGGER: () = env_logger::init();
    }

    #[test]
    fn test_read_geog_lookup() -> anyhow::Result<()> {
        let df = read_geog_lookup(format!("{PERSISTENT_DATA_DIR}/gb_geog_lookup.csv.gz"))?;
        println!("{}", df);
        Ok(())
    }

    #[test]
    fn test_assignment_new() -> anyhow::Result<()> {
        Assignment::new("E09000001", 0, &TEST_CONFIG)?;
        Ok(())
    }

    #[test]
    fn test_assignment_run() -> anyhow::Result<()> {
        // Init env logger
        let _ = &ENV_LOGGER;

        let mut assignment = Assignment::new("E09000001", 0, &TEST_CONFIG)?;
        assignment.run()?;

        // Test only each person only assigned to single household
        let mut counts: HashMap<PID, usize> = HashMap::new();
        assignment
            .h_data
            .iter()
            .flat_map(|hh| hh.hrpid)
            .for_each(|pid| {
                counts.entry(pid).and_modify(|v| *v += 1).or_insert(1);
            });
        let multi_assigned_pids = counts
            .iter()
            .filter_map(|(k, &v)| if v > 1 { Some(*k) } else { None })
            .collect::<Vec<_>>();
        assert_eq!(multi_assigned_pids.len(), 0);

        // Test assigned household PIDs matches people's household HID
        assignment.h_data.iter().for_each(|hh| {
            if let Some(pid) = hh.hrpid {
                let person = assignment.p_data.get(pid).unwrap();
                assert_eq!(person.hid.unwrap(), hh.hid)
            }
        });
        Ok(())
    }

    #[test]
    fn test_assignment_determinism() -> anyhow::Result<()> {
        // Init env logger
        let _ = &ENV_LOGGER;
        let mut assignment_0_0 = Assignment::new("E09000001", 0, &TEST_CONFIG)?;
        assignment_0_0.run()?;
        let mut assignment_0_1 = Assignment::new("E09000001", 0, &TEST_CONFIG)?;
        assignment_0_1.run()?;
        let mut assignment_1_0 = Assignment::new("E09000001", 1, &TEST_CONFIG)?;
        assignment_1_0.run()?;
        let mut assignment_1_1 = Assignment::new("E09000001", 1, &TEST_CONFIG)?;
        assignment_1_1.run()?;

        // Equal seeds used on different runs should generate equal assignments
        assert_eq!(assignment_0_0.digest()?, assignment_0_1.digest()?);
        assert_eq!(assignment_1_0.digest()?, assignment_1_1.digest()?);

        // Different seeds used on different runs should generate different assignments
        assert_ne!(assignment_0_0.digest()?, assignment_1_0.digest()?);
        Ok(())
    }

    const EXPECTED_DIGEST_0: &str =
        "8bd375286de609f387535fe21d290faa3e61551a1e30c6352aed13347df60414";

    #[test]
    #[ignore = "single machine testing: not expected to be valid across platforms as exact digest used"]
    fn test_assignment_determinism_digest() -> anyhow::Result<()> {
        // Init env logger
        let _ = &ENV_LOGGER;
        let mut assignment_0_0 = Assignment::new(
            "E06000001",
            0,
            &Config {
                data_dir: PathBuf::from_str("../data/").unwrap(),
                ..TEST_CONFIG.clone()
            },
        )?;
        assignment_0_0.run()?;
        assert_eq!(assignment_0_0.digest()?, EXPECTED_DIGEST_0);

        Ok(())
    }
}
