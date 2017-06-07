#include <iostream>

int
main() {
  float x = 1.0f;

  while (x > 0.0f) {
    std::cout << x << std::endl;
    x *= 0.5f;
  }

  return 0;
}
