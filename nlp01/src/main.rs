mod forward_net;

use nalgebra::*;
use rand::prelude::*;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn main() {
    let w = Matrix2x3::from_rows(&[RowVector3::new(1, 2, 3), RowVector3::new(4, 5, 6)]);

    let x = Matrix2x3::from_rows(&[RowVector3::new(0, 1, 2), RowVector3::new(3, 4, 5)]);

    println!("{}", w + x);
    println!("{}", w.component_mul(&x));

    let a = Matrix2::new(1, 2, 3, 4);

    println!("{}", a);
    println!("{}", a * 10);

    let a = Vector3::new(1, 2, 3);
    let b = Vector3::new(4, 5, 6);

    println!("{}", a.dot(&b));

    let a = Matrix2::from_rows(&[RowVector2::new(1, 2), RowVector2::new(3, 4)]);
    let b = Matrix2::from_rows(&[RowVector2::new(5, 6), RowVector2::new(7, 8)]);

    println!("{}", a * b);

    let mut rng = thread_rng();

    let w1: SMatrix<f64, 2, 4> =
        SMatrix::from_vec((0..8).into_iter().map(|_| rng.gen::<f64>()).collect());

    println!("w1 = {w1}");

    let b1 = RowVector4::from_vec((0..4).into_iter().map(|_| rng.gen::<f64>()).collect());

    println!("b1 = {b1}");

    let x: SMatrix<f64, 10, 2> =
        SMatrix::from_vec((0..20).into_iter().map(|_| rng.gen::<f64>()).collect());

    println!("x = {x}");

    let b1: SMatrix<f64, 10, 4> = SMatrix::from_rows(
        (0..10)
            .into_iter()
            .map(|_| b1)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    let h = x * w1 + b1;

    println!("h = {h}");

    let w2: SMatrix<f64, 4, 3> =
        SMatrix::from_vec((0..12).into_iter().map(|_| rng.gen::<f64>()).collect());

    let b2 = RowVector3::from_vec((0..3).into_iter().map(|_| rng.gen::<f64>()).collect());

    let h = x * w1 + b1;
    println!("h = {h}");
    let a = h.map(|el| sigmoid(el));
    println!("a = {a}");

    println!("s.shape() = {:?}", (a * w2).shape());

    let b2: SMatrix<f64, 10, 3> = SMatrix::from_rows(
        (0..10)
            .into_iter()
            .map(|_| b2)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    let s = a * w2 + b2;
    println!("s = {}", s);
}
