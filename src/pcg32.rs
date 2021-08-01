/// Some utility functions including random number generation
use rand::distributions::{
    uniform::{SampleRange, SampleUniform},
    Distribution, Standard,
};
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct Pcg32 {
    rng_gen: rand_pcg::Pcg32,
}

impl Pcg32 {
    /// Creates a PCG32 random number generator seeded with 0
    pub fn new() -> Pcg32 {
        Pcg32 {
            rng_gen: rand_pcg::Pcg32::seed_from_u64(0),
        }
    }

    /// Creates a PCG32 random number generator from a specific seed
    pub fn from(seed: u64) -> Pcg32 {
        Pcg32 {
            rng_gen: rand_pcg::Pcg32::seed_from_u64(seed),
        }
    }

    /// Returns the next random number from the generator
    pub fn gen<T>(&mut self) -> T
    where
        Standard: Distribution<T>,
    {
        self.rng_gen.gen::<T>()
    }

    /// Returns a random number in the given range [start,end)
    pub fn _in_range<T, R>(&mut self, range: R) -> T
    where
        T: SampleUniform,
        R: SampleRange<T>,
    {
        self.rng_gen.gen_range::<T, R>(range)
    }
}

impl Default for Pcg32 {
    fn default() -> Self {
        Self::new()
    }
}
