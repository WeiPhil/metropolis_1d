use crate::metropolis::Function;

#[derive(PartialEq)]
pub enum FunctionType {
    ShiftedSquare,
    Sinus,
    None,
}

pub fn shifted_square_and_norm() -> (Box<Function<f32>>, f32) {
    (
        Box::new(|x: f32| -> f32 {
            if (0.0..=1.0).contains(&x) {
                (x - 0.5).powf(2.0)
            } else {
                0.0
            }
        }),
        12.0,
    )
}

pub fn shifted_square_f64(x: f64) -> f64 {
    if (0.0..=1.0).contains(&x) {
        (x - 0.5).powf(2.0)
    } else {
        0.0
    }
}

pub fn sinus_and_norm() -> (Box<Function<f32>>, f32) {
    (
        Box::new(|x: f32| -> f32 {
            if (0.0..=1.0).contains(&x) {
                (10.0 * x).sin().abs()
            } else {
                0.0
            }
        }),
        1.0 / 0.616,
    )
}

pub fn sinus_f64(x: f64) -> f64 {
    if (0.0..=1.0).contains(&x) {
        (10.0 * x).sin().abs()
    } else {
        0.0
    }
}
