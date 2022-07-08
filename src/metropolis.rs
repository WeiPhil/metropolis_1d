use crate::pcg32::Pcg32;

pub type Function<T> = dyn Fn(T) -> T + 'static;

/// A Metropolis sampler that stores a sequence of generated samples
/// and the normalisation factor of the target function.
pub struct Metropolis {
    sample_sequence: Vec<(f32, f32)>,
    f_norm: f32,
}

/// We perform a small mutation with probability `small_mutate_prob` and otherwise a large mutation
fn mutate(prev_sample: f32, pcg_rand: &mut Pcg32, small_mutate_prob: f32) -> f32 {
    if pcg_rand.gen::<f32>() < small_mutate_prob {
        mutate_small(prev_sample, pcg_rand)
    } else {
        mutate_large(prev_sample, pcg_rand)
    }
}

/// A large mutation : X -> U(0,1)
fn mutate_large(_prev_sample: f32, pcg_rand: &mut Pcg32) -> f32 {
    pcg_rand.gen::<f32>()
}

/// A small mutation :  X -> X + U(-0.1,0.1)
fn mutate_small(prev_sample: f32, pcg_rand: &mut Pcg32) -> f32 {
    let mutated = prev_sample + 0.1 * (pcg_rand.gen::<f32>() - 0.5);
    mutated
}

impl Metropolis {
    /// X = X0
    /// for i = 1 to n
    ///      X' = mutate(X)
    ///      a = accept(X, X')
    ///      if expected_value_technique
    ///         record(X, 1 - a)
    ///         record(X', a)
    ///      if (random() < a)
    ///          X = X'
    ///      if !expected_value_technique
    ///         record(X, 1)
    pub fn gen_sample_sequence(
        seed: u64,
        f_and_norm: &(Box<Function<f32>>, f32),
        metropolis_samples: usize,
        burn_in_samples: usize,
        mut small_mutate_prob: f32,
        expected_value_technique: bool,
    ) -> Metropolis {
        let (f, norm) = f_and_norm;
        small_mutate_prob = small_mutate_prob.clamp(0.0, 1.0);

        let mut pcg_rand = Pcg32::from(seed);
        let x0 = pcg_rand.gen::<f32>();
        let mut x = x0;
        let mut sample_sequence = vec![(x0, 1.0)];
        for i in 0..(burn_in_samples + metropolis_samples) {
            let x_prime = mutate(x, &mut pcg_rand, small_mutate_prob);

            let acceptance = (f(x_prime) / f(x)).min(1.0);

            if expected_value_technique && i >= burn_in_samples {
                sample_sequence.push((x, (1.0 - acceptance)));
                sample_sequence.push((x_prime, acceptance));
            }

            if pcg_rand.gen::<f32>() < acceptance {
                x = x_prime;
            }

            if !expected_value_technique && i >= burn_in_samples {
                sample_sequence.push((x, 1.0));
            }
        }

        Metropolis {
            sample_sequence,
            f_norm: *norm,
        }
    }

    /// Returns the distribution of samples in `num_bins` bins (renormalised to match the reference function)
    pub fn sample_distribution(&self, num_bins: usize) -> Vec<f32> {
        let mut bins = self
            .sample_sequence
            .iter()
            .fold(vec![0.0; num_bins], |mut bins, sample| {
                let bin = ((sample.0 * num_bins as f32) as usize).min(num_bins - 1);
                bins[bin] += sample.1;
                bins
            });

        let normalisation = bins.iter().sum::<f32>() / (num_bins as f32) * self.f_norm;
        bins.iter_mut().for_each(|x| *x /= normalisation);
        bins
    }
}
