#include <bits/stdc++.h>
using namespace std;

float randfl() {
  return ((double)rand()) / RAND_MAX;
}

int main() {
  puts("2");

  puts("0");
  puts("10 10");
  for (int i = 0; i < 10 * 10; ++i)
    printf("1.0 1.0 1.0 %f %f %f 2 2 3.0 0\n", randfl(), randfl(), randfl());

  puts("100");
  for (int i = 0; i < 100; ++i)
    printf("1.0 2.0 %d 0.1 0.2 0.3 4 5 6.0\n", i);

  puts("1 1");
  printf("0.5 0.5 0.5 1.0 2.0 0.0 1 2 3.0 50");
  for (int i = 49; i >= 0; --i)
    printf(" %d", i);
  puts("");
  return 0;
}
