// Loads add.pte, runs forward(2.0, 3.0), expects 5.0. No JVM involved.
#include <cstdio>
#include <vector>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::Module;
using executorch::extension::from_blob;

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  Module module(pte);

  std::vector<float> a{2.0f};
  std::vector<float> b{3.0f};
  auto ta = from_blob(a.data(), {1});
  auto tb = from_blob(b.data(), {1});

  auto result = module.forward({ta, tb});
  if (!result.ok()) {
    std::fprintf(stderr, "forward failed, error=%d\n",
                 static_cast<int>(result.error()));
    return 1;
  }
  float out = result->at(0).toTensor().const_data_ptr<float>()[0];
  std::printf("RESULT=%.1f\n", out);
  return (out == 5.0f) ? 0 : 2;
}