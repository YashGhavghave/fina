use pyo3::prelude::*;

#[pyfunction]
fn mean(data: Vec<f64>) -> PyResult<f64> {
    if data.is_empty() {
        Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"))
    } else {
        Ok(data.iter().sum::<f64>() / data.len() as f64)
    }
}

#[pyfunction]
fn variance(data: Vec<f64>) -> PyResult<f64> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    let m = mean(data.clone())?;
    Ok(data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64)
}

#[pyfunction]
fn std_dev(data: Vec<f64>) -> PyResult<f64> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    Ok(variance(data)?.sqrt())
}

#[pyfunction]
fn dot(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        Err(pyo3::exceptions::PyValueError::new_err("Vectors must be same length"))
    } else {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }
}

#[pyfunction]
fn euclidean(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        Err(pyo3::exceptions::PyValueError::new_err("Vectors must be same length"))
    } else {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt())
    }
}

#[pyfunction]
fn sigmoid(x: f64) -> PyResult<f64> {
    // Handle extreme values to prevent overflow
    if x > 500.0 {
        Ok(1.0)
    } else if x < -500.0 {
        Ok(0.0)
    } else {
        Ok(1.0 / (1.0 + (-x).exp()))
    }
}

#[pyfunction]
fn relu(x: f64) -> PyResult<f64> {
    Ok(x.max(0.0))
}

#[pyfunction]
fn softmax(data: Vec<f64>) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    
    // Subtract max for numerical stability
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp = exp_values.iter().sum::<f64>();
    
    if sum_exp == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Underflow in softmax computation"));
    }
    
    Ok(exp_values.iter().map(|&x| x / sum_exp).collect())
}

#[pyfunction]
fn cross_entropy(pred: Vec<f64>, target: Vec<f64>) -> PyResult<f64> {
    if pred.len() != target.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Vectors must be same length"));
    }
    if pred.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Vectors cannot be empty"));
    }
    
    let mut loss = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        if *p <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Predictions must be positive for cross entropy"));
        }
        loss += t * p.ln();
    }
    Ok(-loss)
}

#[pyfunction]
fn mse(pred: Vec<f64>, target: Vec<f64>) -> PyResult<f64> {
    if pred.len() != target.len() {
        Err(pyo3::exceptions::PyValueError::new_err("Vectors must be same length"))
    } else if pred.is_empty() {
        Err(pyo3::exceptions::PyValueError::new_err("Vectors cannot be empty"))
    } else {
        Ok(pred.iter().zip(target.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64)
    }
}

#[pyfunction]
fn min_max_normalize(data: Vec<f64>) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    if (max_val - min_val).abs() < f64::EPSILON {
        Err(pyo3::exceptions::PyValueError::new_err("All elements are equal, cannot normalize"))
    } else {
        Ok(data.iter().map(|&x| (x - min_val) / (max_val - min_val)).collect())
    }
}

#[pyfunction]
fn z_score_normalize(data: Vec<f64>) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    
    let m = mean(data.clone())?;
    let s = std_dev(data.clone())?;
    
    if s.abs() < f64::EPSILON {
        return Err(pyo3::exceptions::PyValueError::new_err("Standard deviation is zero, cannot normalize"));
    }
    
    Ok(data.iter().map(|&x| (x - m) / s).collect())
}

#[pyfunction]
fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Vectors must be same length"));
    }
    if a.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Vectors cannot be empty"));
    }
    
    let dot_ab = dot(a.clone(), b.clone())?;
    let norm_a_squared = dot(a.clone(), a.clone())?;
    let norm_b_squared = dot(b.clone(), b.clone())?;
    let norm_a = norm_a_squared.sqrt();
    let norm_b = norm_b_squared.sqrt();
    
    if norm_a.abs() < f64::EPSILON || norm_b.abs() < f64::EPSILON {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot compute cosine similarity for zero vectors"));
    }
    
    Ok(dot_ab / (norm_a * norm_b))
}

#[pyfunction]
fn log_loss(pred: Vec<f64>, target: Vec<f64>) -> PyResult<f64> {
    if pred.len() != target.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Vectors must be same length"));
    }
    if pred.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Vectors cannot be empty"));
    }
    
    let mut loss = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        // Clamp predictions to avoid log(0)
        let p_clamped = p.max(f64::EPSILON).min(1.0 - f64::EPSILON);
        loss += t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln();
    }
    Ok(-loss / pred.len() as f64)
}

#[pyfunction]
fn ema(data: Vec<f64>, alpha: f64) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(pyo3::exceptions::PyValueError::new_err("Alpha must be between 0 and 1"));
    }
    
    let mut result = Vec::with_capacity(data.len());
    let mut ema_value = data[0];
    result.push(ema_value);
    
    for &x in &data[1..] {
        ema_value = alpha * x + (1.0 - alpha) * ema_value;
        result.push(ema_value);
    }
    Ok(result)
}

#[pyfunction]
fn tanh_activation(x: f64) -> PyResult<f64> {
    Ok(x.tanh())
}

#[pyfunction]
fn leaky_relu(x: f64, alpha: f64) -> PyResult<f64> {
    Ok(if x >= 0.0 { x } else { alpha * x })
}

#[pyfunction]
fn rms(data: Vec<f64>) -> PyResult<f64> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    Ok((data.iter().map(|x| x.powi(2)).sum::<f64>() / data.len() as f64).sqrt())
}

#[pyfunction]
fn clamp(x: f64, min_val: f64, max_val: f64) -> PyResult<f64> {
    if min_val > max_val {
        return Err(pyo3::exceptions::PyValueError::new_err("min_val cannot be greater than max_val"));
    }
    Ok(x.clamp(min_val, max_val))
}

#[pymodule]
fn fina(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(mse, m)?)?;
    m.add_function(wrap_pyfunction!(min_max_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(z_score_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(log_loss, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(tanh_activation, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(rms, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    Ok(())
}