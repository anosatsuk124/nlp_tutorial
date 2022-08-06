use nalgebra::DMatrix;

trait Layer<T> {
    fn forward(&self, x: DMatrix<T>) -> DMatrix<T>;
}

#[derive(Default)]
struct Sigmoid {
    params: Vec<DMatrix<f64>>,
}

impl Sigmoid {
    fn new() -> Self {
        Self { params: Vec::new() }
    }
}

impl Layer<f64> for Sigmoid {
    fn forward(&self, x: DMatrix<f64>) -> DMatrix<f64> {
        x.map(|x| 1. / (1. + (-x).exp()))
    }
}

struct Affine {
    params: Vec<DMatrix<f64>>,
}

impl Affine {
    fn new(w: DMatrix<f64>, b: DMatrix<f64>) -> Self {
        Self { params: vec![w, b] }
    }
}

impl Layer<f64> for Affine {
    fn forward(&self, x: DMatrix<f64>) -> DMatrix<f64> {
        let w = &self.params[0];
        let b = &self.params[1];
        x * w + b
    }
}
