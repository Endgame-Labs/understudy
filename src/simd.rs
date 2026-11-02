//! The few f32 kernels the readouts spend their time in. Each is written once over 16 independent lanes (so the
//! summation order, and the result, is the same on every ISA) and instantiated for AVX-512, AVX2 and the baseline,
//! picked at runtime: eg2cpu's approach, with no `target-cpu` needed at build time.

use std::sync::OnceLock;

const L: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Isa {
    Avx512,
    Avx2,
    Base,
}

fn isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Isa::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return Isa::Avx2;
            }
        }
        Isa::Base
    })
}

#[inline(always)]
fn dot_k(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0f32; L];
    let (ca, cb) = (a[..n].chunks_exact(L), b[..n].chunks_exact(L));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..L {
            acc[l] += x[l] * y[l];
        }
    }
    for (l, (x, y)) in ra.iter().zip(rb).enumerate() {
        acc[l] += x * y;
    }
    hsum(&acc)
}

#[inline(always)]
fn hsum(acc: &[f32; L]) -> f32 {
    let mut s = [0f32; 8];
    for l in 0..8 {
        s[l] = acc[l] + acc[l + 8];
    }
    ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))
}

#[inline(always)]
fn axpy_k(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// Four dot products of `h` with `q[k]` in one pass over `h`.
#[inline(always)]
fn dot4_k(h: &[f32], q: [&[f32]; 4]) -> [f32; 4] {
    let n = h.len();
    let mut acc = [[0f32; L]; 4];
    let full = n / L * L;
    let mut i = 0;
    while i < full {
        let x = &h[i..i + L];
        for k in 0..4 {
            let qk = &q[k][i..i + L];
            for l in 0..L {
                acc[k][l] += x[l] * qk[l];
            }
        }
        i += L;
    }
    for j in full..n {
        for k in 0..4 {
            acc[k][j - full] += h[j] * q[k][j];
        }
    }
    [hsum(&acc[0]), hsum(&acc[1]), hsum(&acc[2]), hsum(&acc[3])]
}

/// `y[k] += alpha[k] * h` for four outputs in one pass over `h`.
#[inline(always)]
fn axpy4_k(alpha: [f32; 4], h: &[f32], y: [&mut [f32]; 4]) {
    let [y0, y1, y2, y3] = y;
    for (j, &x) in h.iter().enumerate() {
        y0[j] += alpha[0] * x;
        y1[j] += alpha[1] * x;
        y2[j] += alpha[2] * x;
        y3[j] += alpha[3] * x;
    }
}

macro_rules! dispatch {
    ($name:ident, $kernel:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        pub fn $name($($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx512f,avx2,fma")]
                unsafe fn avx512($($arg: $ty),*) -> $ret { $kernel($($arg),*) }
                #[target_feature(enable = "avx2,fma")]
                unsafe fn avx2($($arg: $ty),*) -> $ret { $kernel($($arg),*) }
                match isa() {
                    // SAFETY: the features were detected at runtime.
                    Isa::Avx512 => return unsafe { avx512($($arg),*) },
                    Isa::Avx2 => return unsafe { avx2($($arg),*) },
                    Isa::Base => {}
                }
            }
            $kernel($($arg),*)
        }
    };
}

dispatch!(dot, dot_k, (a: &[f32], b: &[f32]) -> f32);
dispatch!(axpy, axpy_k, (alpha: f32, x: &[f32], y: &mut [f32]) -> ());
dispatch!(dot4, dot4_k, (h: &[f32], q: [&[f32]; 4]) -> [f32; 4]);
dispatch!(axpy4, axpy4_k, (alpha: [f32; 4], h: &[f32], y: [&mut [f32]; 4]) -> ());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.1).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - want).abs() < 1e-3);
        let d = dot4(&a, [&b, &a, &b, &a]);
        assert!((d[0] - want).abs() < 1e-3 && (d[1] - dot(&a, &a)).abs() < 1e-3);
        let mut y = vec![1f32; 37];
        axpy(2.0, &a, &mut y);
        assert_eq!(y[3], 1.0 + 2.0 * 1.5);
    }
}
