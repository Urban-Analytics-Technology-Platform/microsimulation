use serde::{Deserialize, Serialize};

use crate::{person::PID, Eth, OA};

type UInt = u32;
type Int = i32;
#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct HID(pub usize);
impl std::fmt::Display for HID {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "HID #{}", self.0)
    }
}
impl From<usize> for HID {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

pub fn serialize_bool<S>(bool: &Option<bool>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    let s = match bool {
        Some(true) => "True",
        Some(false) => "False",
        None => "False",
    };
    serializer.serialize_str(s)
}

pub fn serialize_hrpid<S>(pid: &Option<PID>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    let s = match pid {
        Some(pid) => format!("{}", pid.0),
        None => "-1".to_string(),
    };
    serializer.serialize_str(&s)
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct Household {
    #[serde(rename = "HID")]
    pub hid: HID,
    #[serde(rename = "Area")]
    // TODO: should be OA? Use import of wrapper types instead.
    pub oa: OA,
    #[serde(rename = "LC4402_C_TYPACCOM")]
    pub lc4402_c_typaccom: Int,
    #[serde(rename = "QS420_CELL")]
    // TODO check
    pub qs420_cell: Int,
    #[serde(rename = "LC4402_C_TENHUK11")]
    pub lc4402_c_tenhuk11: Int,
    #[serde(rename = "LC4408_C_AHTHUK11")]
    pub lc4408_c_ahthuk11: Int,
    #[serde(rename = "CommunalSize")]
    pub communal_size: Int,
    #[serde(rename = "LC4404_C_SIZHUK11")]
    pub lc4404_c_sizhuk11: Int,
    #[serde(rename = "LC4404_C_ROOMS")]
    pub lc4404_c_rooms: Int,
    #[serde(rename = "LC4405EW_C_BEDROOMS")]
    pub lc4405ew_c_bedrooms: Int,
    #[serde(rename = "LC4408EW_C_PPBROOMHEW11")]
    pub lc4408ew_c_ppbroomhew11: Int,
    #[serde(rename = "LC4402_C_CENHEATHUK11")]
    pub lc4402_c_cenheathuk11: UInt,
    #[serde(rename = "LC4605_C_NSSEC")]
    pub lc4605_c_nssec: Int,
    #[serde(rename = "LC4202_C_ETHHUK11")]
    pub lc4202_c_ethhuk11: Eth,
    #[serde(rename = "LC4202_C_CARSNO")]
    pub lc4202_c_carsno: UInt,
    #[serde(rename = "HRPID")]
    #[serde(serialize_with = "serialize_hrpid")]
    pub hrpid: Option<PID>,
    #[serde(rename = "FILLED")]
    #[serde(serialize_with = "serialize_bool")]
    pub filled: Option<bool>,
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_deserialize_hh() -> anyhow::Result<()> {
        let test_csv = std::fs::read_to_string("tests/data/ssm_hh_E09000001_OA11_2020.csv")?;
        let mut rdr = csv::Reader::from_reader(test_csv.as_bytes());
        for result in rdr.deserialize() {
            let record: Household = result?;
            println!("{:?}", record);
        }
        Ok(())
    }
}
