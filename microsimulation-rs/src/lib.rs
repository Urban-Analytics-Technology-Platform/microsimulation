use std::ops::Deref;

use serde::{Deserialize, Serialize};

pub mod assignment;
pub mod config;
pub mod household;
pub mod person;
pub(crate) mod queues;

use sha2::{Digest, Sha256};

const ADULT_AGE: Age = Age(16);

fn digest(object: impl Serialize) -> anyhow::Result<String> {
    Ok(hex::encode(Sha256::digest(bincode::serialize(&object)?)))
}

#[derive(Hash, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct MSOA(String);

impl std::fmt::Display for MSOA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for MSOA {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}
impl From<String> for MSOA {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<MSOA> for String {
    fn from(value: MSOA) -> Self {
        value.0
    }
}

impl Deref for MSOA {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
// TODO: use type instead of string in assignment
#[derive(Hash, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct OA(String);

impl From<&str> for OA {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}
impl From<String> for OA {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Hash, Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sex(pub usize);

impl Sex {
    fn opposite(&self) -> Self {
        Self(3 - self.0)
    }
}

impl std::fmt::Display for Sex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sex: {}", self.0)
    }
}

#[derive(Hash, Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Age(pub usize);

impl Ord for Age {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Age {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for Age {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Age: {}", self.0)
    }
}

#[derive(Hash, Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Eth(pub i32);

#[derive(Hash, Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct EthEW(pub i32);

impl From<i32> for Eth {
    fn from(value: i32) -> Self {
        Self(value)
    }
}

impl From<EthEW> for Eth {
    fn from(value: EthEW) -> Self {
        Self(value.0)
    }
}

impl std::fmt::Display for Eth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Eth: {}", self.0)
    }
}

impl From<i32> for EthEW {
    fn from(value: i32) -> Self {
        Self(value)
    }
}
