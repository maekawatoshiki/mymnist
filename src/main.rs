use std::cmp::Ordering;

fn main() {
    let input = vec![
        vec![
            include!("input2") // seven
                .into_iter()
                .map(|x| x.into_iter().map(|y| y as f32 / 255.0).collect())
                .collect();
            1
        ];
        1
    ]; // 1x1x28x28
    let conv_ = conv(input, include!("conv1"));
    let add_ = add(conv_, include!("add1"));
    let relu_ = relu(add_);
    let maxpool_ = maxpool(relu_, 2, 2);
    let conv_ = conv(maxpool_, include!("conv2"));
    let add_ = add(conv_, include!("add2"));
    let relu_ = relu(add_);
    let maxpool_ = maxpool(relu_, 3, 3);
    let reshape_ = reshape1x256(maxpool_);
    let reshape1_ = reshape256x10(include!("reshape1"));
    let matmal_ = matmal(reshape_, reshape1_);
    let add_ = add2(matmal_, include!("add3"));
    println!(
        "inferred: {}",
        add_[0]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap(),
    );
}

type Tensor4D = Vec<Vec<Vec<Vec<f32>>>>;
type Tensor3D = Vec<Vec<Vec<f32>>>;
type Tensor2D = Vec<Vec<f32>>;

#[allow(non_snake_case)]
fn matmal(A: Tensor2D, B: Tensor2D) -> Tensor2D {
    let mut out = Tensor2D::new();
    for a in 0..A.len() {
        assert_eq!(A[0].len(), B.len());
        let mut out2 = vec![];
        for c in 0..B[0].len() {
            let mut t = 0.0;
            for b in 0..B.len() {
                t += A[a][b] * B[b][c];
            }
            out2.push(t);
        }
        out.push(out2);
    }
    out
}

#[allow(non_snake_case)]
fn reshape1x256(X: Tensor4D) -> Tensor2D {
    let mut out_n = vec![];
    for n in 0..X.len() {
        let mut out_c = vec![];
        for c in 0..X[n].len() {
            for h in 0..X[n][c].len() {
                for w in 0..X[n][c][h].len() {
                    out_c.push(X[n][c][h][w]);
                }
            }
        }
        out_n.push(out_c);
    }
    out_n
}

#[allow(non_snake_case)]
fn reshape256x10(X: Tensor4D) -> Tensor2D {
    let mut out_n = vec![];
    for n in 0..X.len() {
        for c in 0..X[n].len() {
            for h in 0..X[n][c].len() {
                let mut out_w = vec![];
                for w in 0..X[n][c][h].len() {
                    out_w.push(X[n][c][h][w]);
                }
                out_n.push(out_w);
            }
        }
    }
    out_n
}

#[allow(non_snake_case)]
fn maxpool(X: Tensor4D, kernel: usize, stride: usize) -> Tensor4D {
    let dilation = 0;
    let mut out_n = vec![];
    for n in 0..X.len() {
        let mut out_c = vec![];
        for c in 0..X[n].len() {
            let mut out_h = vec![];
            for h in (0..X[n][c].len()).step_by(stride) {
                let mut out_w = vec![];
                'a: for w in (0..X[n][c][h].len()).step_by(stride) {
                    let mut max = f32::MIN;
                    for hh in (h..h + (dilation + 1) * kernel).step_by(dilation + 1) {
                        for ww in (w..w + (dilation + 1) * kernel).step_by(dilation + 1) {
                            if hh >= X[n][c].len() || ww >= X[n][c][0].len() {
                                break 'a;
                            }
                            if max < X[n][c][hh][ww] {
                                max = X[n][c][hh][ww]
                            }
                        }
                    }
                    out_w.push(max);
                }
                out_h.push(out_w);
            }
            out_c.push(out_h);
        }
        out_n.push(out_c);
    }
    out_n
}

#[allow(non_snake_case)]
fn relu(X: Tensor4D) -> Tensor4D {
    let mut out_n = vec![];
    for n in 0..X.len() {
        let mut out_c = vec![];
        for c in 0..X[n].len() {
            let mut out_h = vec![];
            for h in 0..X[n][c].len() {
                let mut out_w = vec![];
                for w in 0..X[n][c][h].len() {
                    out_w.push(if X[n][c][h][w] < 0.0 {
                        0.0
                    } else {
                        X[n][c][h][w]
                    })
                }
                out_h.push(out_w);
            }
            out_c.push(out_h);
        }
        out_n.push(out_c);
    }
    out_n
}

#[allow(non_snake_case)]
fn add(A: Tensor4D, B: Tensor3D) -> Tensor4D {
    let mut out_n = vec![];
    for n in 0..A.len() {
        let mut out_c = vec![];
        assert_eq!(A[n].len(), B.len());
        for c in 0..A[n].len() {
            let mut out_h = vec![];
            for h in 0..A[n][c].len() {
                let mut out_w = vec![];
                for w in 0..A[n][c][h].len() {
                    assert_eq!(B[c].len(), 1);
                    assert_eq!(B[c][0].len(), 1);
                    out_w.push(A[n][c][h][w] + B[c][0][0]);
                }
                out_h.push(out_w);
            }
            out_c.push(out_h);
        }
        out_n.push(out_c);
    }
    out_n
}

#[allow(non_snake_case)]
fn add2(A: Tensor2D, B: Tensor2D) -> Tensor2D {
    let mut output = vec![];
    for _a in 0..A.len() {
        assert_eq!(A.len(), B.len());
        let mut out = vec![];
        for (b, c) in A[0].iter().zip(B[0].iter()) {
            out.push(b + c);
        }
        output.push(out)
    }
    output
}

#[allow(non_snake_case)]
fn conv(X: Tensor4D, W: Tensor4D) -> Tensor4D {
    // output: n m h w
    let mut out_n = vec![];
    for n in 0..X.len() {
        ////////////////////
        let mut out_m = vec![];
        for m in 0..W.len() {
            assert_eq!(X[n].len(), W[m].len());
            // for cg in 0..W[m].len() {
            ///////////////
            let mut out_hw = vec![];
            for h in 0..X[n][0].len() {
                let mut out_w = vec![];
                for w in 0..X[n][0][h].len() {
                    let mut out = 0.0;
                    for c in 0..X[n].len() {
                        let kh_half = W[m][c].len() / 2;
                        for kh in 0..W[m][c].len() {
                            let kw_half = W[m][c][kh].len() / 2;
                            for kw in 0..W[m][c][kh].len() {
                                let h_guard = h as isize - kh_half as isize + kh as isize;
                                let w_guard = w as isize - kw_half as isize + kw as isize;
                                let v = if h_guard < 0
                                    || h_guard >= X[n][c].len() as isize
                                    || w_guard < 0
                                    || w_guard >= X[n][c][0].len() as isize
                                {
                                    0.0
                                } else {
                                    X[n][c][h_guard as usize][w_guard as usize] * W[m][c][kh][kw]
                                };
                                out += v;
                            }
                        }
                    }
                    out_w.push(out);
                }
                out_hw.push(out_w);
            }
            out_m.push(out_hw);
        }
        out_n.push(out_m);
    }
    out_n
}
