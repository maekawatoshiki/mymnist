#include <assert.h>
#include <blis.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double now_in_sec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000.f / 1000.f;
}

void conv_Convolution28(float out[1][8][28][28],
                               const float in[1][1][28][28]) {
  const float weight[8][1][5][5] = {
#include "Convolution28.weight"
  };

  for (int n = 0; n < 1; n++) {
    for (int oc = 0; oc < 8; oc++) {
      for (int h = 0; h < 28; h++) {
        for (int w = 0; w < 28; w++) {
          float sum = 0.f;
          for (int ic = 0; ic < 1; ic++) {
            for (int kh = 0; kh < 5; kh++) {
              for (int kw = 0; kw < 5; kw++) {
                const int h_guard = h - 5 / 2 + kh;
                const int w_guard = w - 5 / 2 + kw;
                float v;
                if (h_guard < 0 || h_guard >= 28 || w_guard < 0 ||
                    w_guard >= 28) {
                  v = 0.f;
                } else {
                  v = in[n][ic][h_guard][w_guard] * weight[oc][ic][kh][kw];
                }
                sum += v;
              }
            }
          }
          out[n][oc][h][w] = sum;
        }
      }
    }
  }
}

void add_Plus30(float out[1][8][28][28], const float in[1][8][28][28]) {
  const float consts[8] = {-0.1615397185087204,  -0.4338356554508209,
                           0.09164135903120041,  -0.01685221679508686,
                           -0.06502643972635269, -0.1317378729581833,
                           0.020417550578713417, -0.1211102306842804};
  for (int n0 = 0; n0 < 1; n0++) {
    for (int n1 = 0; n1 < 8; n1++) {
      for (int n2 = 0; n2 < 28; n2++) {
        for (int n3 = 0; n3 < 28; n3++) {
          out[n0][n1][n2][n3] = in[n0][n1][n2][n3] + consts[n1];
        }
      }
    }
  }
}

void relu_ReLU32(float out[1][8][28][28], const float in[1][8][28][28]) {
  float *out_ptr = (float *)out, *in_ptr = (float *)in;
  for (int i = 0; i < 1 * 8 * 28 * 28; i++) {
    *out_ptr = *in_ptr > 0.f ? *in_ptr : 0.f;
    in_ptr++;
    out_ptr++;
  }
}

void maxpool_Pooling66(float out[1][8][14][14], const float in[1][8][28][28]) {
  for (int n = 0; n < 1; n++) {
    for (int c = 0; c < 8; c++) {
      int y = 0;
      for (int ah = 0; ah < 28 / 2; ah++) {
        int x = 0;
        for (int aw = 0; aw < 28 / 2; aw++) {
          float max = -10000.f;
          for (int kh = 0; kh < 2; kh++) {
            int iy = y + kh;
            for (int kw = 0; kw < 2; kw++) {
              int ix = x + kw;
              if (ix < 0 || iy < 0 || ix >= 28 || iy >= 28) {
                continue;
              }
              if (max < in[n][c][iy][ix])
                max = in[n][c][iy][ix];
            }
          }
          out[n][c][ah][aw] = max == -10000.f ? 0.f : max;
          x += 2;
        }
        y += 2;
      }
    }
  }
}

void conv_Convolution110(float out[1][16][14][14],
                         const float in[1][8][14][14]) {
  const float weight[16][8][5][5] = {
#include "Convolution110.weight"
  };

  for (int n = 0; n < 1; n++) {
    for (int oc = 0; oc < 16; oc++) {
      for (int h = 0; h < 14; h++) {
        for (int w = 0; w < 14; w++) {
          float sum = 0.f;
          for (int ic = 0; ic < 8; ic++) {
            for (int kh = 0; kh < 5; kh++) {
              for (int kw = 0; kw < 5; kw++) {
                const int h_guard = h - 5 / 2 + kh;
                const int w_guard = w - 5 / 2 + kw;
                float v;
                if (h_guard < 0 || h_guard >= 14 || w_guard < 0 ||
                    w_guard >= 14) {
                  v = 0.f;
                } else {
                  v = in[n][ic][h_guard][w_guard] * weight[oc][ic][kh][kw];
                }
                sum += v;
              }
            }
          }
          out[n][oc][h][w] = sum;
        }
      }
    }
  }
}

void add_Plus112(float out[1][16][14][14], const float in[1][16][14][14]) {
  const float consts[16] = {
      -0.08224882185459137,  -0.10886877775192261, -0.14103959500789642,
      -0.20486916601657867,  -0.17913565039634705, -0.2154383808374405,
      -0.1338050663471222,   -0.19572456181049347, -0.26825064420700073,
      -0.25821220874786377,  -0.07615606486797333, 0.01328414585441351,
      -0.004444644320756197, -0.41474083065986633, -0.17879115045070648,
      -0.03865588828921318};
  for (int n0 = 0; n0 < 1; n0++) {
    for (int n1 = 0; n1 < 16; n1++) {
      for (int n2 = 0; n2 < 14; n2++) {
        for (int n3 = 0; n3 < 14; n3++) {
          out[n0][n1][n2][n3] = in[n0][n1][n2][n3] + consts[n1];
        }
      }
    }
  }
}

void relu_ReLU114(float out[1][16][14][14], const float in[1][16][14][14]) {
  float *out_ptr = (float *)out, *in_ptr = (float *)in;
  for (int i = 0; i < 1 * 16 * 14 * 14; i++) {
    *out_ptr = *in_ptr > 0.f ? *in_ptr : 0.f;
    in_ptr++;
    out_ptr++;
  }
}

void maxpool_Pooling160(float out[1][16][4][4], const float in[1][16][14][14]) {
  for (int n = 0; n < 1; n++) {
    for (int c = 0; c < 16; c++) {
      int y = 0;
      for (int ah = 0; ah < 4; ah++) {
        int x = 0;
        for (int aw = 0; aw < 4; aw++) {
          float max = -10000.f;
          for (int kh = 0; kh < 3; kh++) {
            int iy = y + kh;
            for (int kw = 0; kw < 3; kw++) {
              int ix = x + kw;
              if (ix < 0 || iy < 0 || ix >= 14 || iy >= 14) {
                continue;
              }
              if (max < in[n][c][iy][ix])
                max = in[n][c][iy][ix];
            }
          }
          out[n][c][ah][aw] = max == -10000.f ? 0.f : max;
          x += 3;
        }
        y += 3;
      }
    }
  }
}

void matmul_Times212(float out[1][10], const float in0[1][256],
                     const float in1[256][10]) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 10, 256, 1.f,
              (float *)in0, 256, (float *)in1, 10, 0.f, (float *)out, 10);
}

void add_Plus214(float out[1][10], const float in[1][10]) {
  const float consts[10] = {-0.04485602676868439,  0.007791661191731691,
                            0.06810081750154495,   0.02999374084174633,
                            -0.1264096349477768,   0.14021874964237213,
                            -0.055284902453422546, -0.04938381537795067,
                            0.08432205021381378,   -0.05454041436314583};
  for (int n0 = 0; n0 < 1; n0++) {
    for (int n1 = 0; n1 < 10; n1++) {
      out[n0][n1] = in[n0][n1] + consts[n1];
    }
  }
}

int main() {
  float *input = malloc(sizeof(float) * 1 * 1 * 28 * 28);
  float *Convolution28_Output_0 = malloc(sizeof(float) * 1 * 8 * 28 * 28);
  float *Plus30_Output_0 = malloc(sizeof(float) * 1 * 8 * 28 * 28);
  float *ReLU32_Output_0 = malloc(sizeof(float) * 1 * 8 * 28 * 28);
  float *Pooling66_Output_0 = malloc(sizeof(float) * 1 * 8 * 14 * 14);
  float *Convolution110_Output_0 = malloc(sizeof(float) * 1 * 16 * 14 * 14);
  float *Plus112_Output_0 = malloc(sizeof(float) * 1 * 16 * 14 * 14);
  float *ReLU114_Output_0 = malloc(sizeof(float) * 1 * 16 * 14 * 14);
  float *Pooling160_Output_0 = malloc(sizeof(float) * 1 * 16 * 4 * 4);
  float *Times212_Output_0 = malloc(sizeof(float) * 1 * 10);
  float *Plus214_Output_0 = malloc(sizeof(float) * 1 * 10);

  const float Parameter193_reshape1[256][10] = {
#include "Times212_reshape1"
  };

  const int test[10 * (1 + 28 * 28)] = {
#include "mnist_test"
  };

  double sum_elapsed = 0.f;
  int n = 0;

  for (int loop = 0; loop < 1000; loop++) {
    for (int n_test = 0; n_test < 10; n_test++) {
      int label = test[n_test * (1 + 28 * 28)];
      int *input_int = &test[n_test * (1 + 28 * 28) + 1];

      double start = now_in_sec();

      for (int i = 0; i < 28 * 28; i++)
        input[i] = ((float)input_int[i]) / 255.f;

      conv_Convolution28(Convolution28_Output_0, input);
      add_Plus30(Plus30_Output_0, Convolution28_Output_0);
      relu_ReLU32(ReLU32_Output_0, Plus30_Output_0);
      maxpool_Pooling66(Pooling66_Output_0, ReLU32_Output_0);
      conv_Convolution110(Convolution110_Output_0, Pooling66_Output_0);
      add_Plus112(Plus112_Output_0, Convolution110_Output_0);
      relu_ReLU114(ReLU114_Output_0, Plus112_Output_0);
      maxpool_Pooling160(Pooling160_Output_0, ReLU114_Output_0);
      matmul_Times212(Times212_Output_0, Pooling160_Output_0,
                      Parameter193_reshape1);
      add_Plus214(Plus214_Output_0, Times212_Output_0);

      double end = now_in_sec();

      sum_elapsed += end - start;
      n++;

      printf("%lf [ms]\n", (sum_elapsed / n) * 1000.f);

      float max = -1000.f;
      int idx = -1;
      for (int i = 0; i < 10; i++) {
        if (max < Plus214_Output_0[i]) {
          max = Plus214_Output_0[i];
          idx = i;
        }
        /* printf("%lf ", Plus214_Output_0[i]); */
      }
      /* puts(""); */
      /* printf("%d label:%d\n", idx, label); */
      /* assert(idx == label); */
    }
  }

  // 0.25ms

  return 0;
}
