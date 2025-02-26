#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <vector>

// 测试配置参数
constexpr int MAIN_LOOPS = 500;   // 主循环次数
constexpr int LINKLIST_SIZE = 10; // 链表测试数据量
constexpr int MATRIX_SIZE = 4;    // 矩阵尺寸
constexpr int NN_INPUTS = 10;     // 神经网络输入数量

// 测试函数原型声明
void test_linked_list();
void test_state_machine();
void test_matrix_compute();
void test_neural_activation();

// 测试调度框架
class BranchTestScheduler {
public:
  using TestFunction = void (*)();

  BranchTestScheduler() : rng(std::random_device{}()) {
    tests = {// 注册所有测试
             &test_linked_list, &test_matrix_compute, &test_neural_activation};
  }

  void run(int total_loops) {
    for (int loop = 1; loop <= total_loops; ++loop) {
      std::cout << "=== Loop " << loop << "/" << total_loops
                << " ===" << std::endl;

      // 生成随机执行顺序
      std::shuffle(tests.begin(), tests.end(), rng);

      // 执行所有测试
      for (auto &test : tests) {
        test();
        // 根据测试函数地址打印完成名称
        if (test == &test_linked_list) {
          std::cout << "test_linked_list completed\n";
        } else if (test == &test_matrix_compute) {
          std::cout << "test_matrix_compute completed\n";
        } else if (test == &test_neural_activation) {
          std::cout << "test_neural_activation completed\n";
        }
      }
    }
  }

private:
  std::vector<TestFunction> tests;
  std::mt19937 rng;
};

//============== 测试实现 ==============

void test_linked_list() {
  struct Node {
    int value;
    Node *next;
    bool flag;
  };

  std::vector<Node> nodes(LINKLIST_SIZE);
  std::mt19937 rng(std::random_device{}());
  std::bernoulli_distribution dist(0.5);

  // 初始化链表
  for (int i = 0; i < LINKLIST_SIZE - 1; ++i) {
    nodes[i].next = &nodes[i + 1];
    nodes[i].flag = dist(rng);
    nodes[i].value = i % 256;
  }

  volatile int dummy = 0;
  Node *current = &nodes[0];
  while (current) {
    if (current->flag) {
      dummy += current->value;
    }
    current = current->next;
  }
}

void test_state_machine() {
  enum class State { A, B, C, D };
  State state = State::A;
  int counter = 0;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 99);

  for (int i = 0; i < 1e7; ++i) {
    switch (state) {
    case State::A:
      if (dist(rng) < 30)
        state = State::B;
      else if (dist(rng) < 60)
        state = State::C;
      break;
    case State::B:
      if (dist(rng) < 20)
        state = State::D;
      else
        state = State::A;
      break;
    case State::C:
      state = (dist(rng) < 50) ? State::A : State::D;
      break;
    case State::D:
      state = (dist(rng) < 40) ? State::B : State::C;
      break;
    }

    if (counter % 128 < 64) {
      counter += 2;
    } else {
      counter += (dist(rng) % 3) - 1;
    }
  }

  volatile int result = counter;
}

void test_matrix_compute() {
  std::vector<std::vector<float>> matrix(MATRIX_SIZE,
                                         std::vector<float>(MATRIX_SIZE));
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &row : matrix) {
    for (auto &val : row) {
      val = dist(rng);
    }
  }

  volatile float result = 0;
  for (int i = 1; i < MATRIX_SIZE - 1; ++i) {
    for (int j = 1; j < MATRIX_SIZE - 1; ++j) {
      if (matrix[i][j] > 0.8f) {
        result += std::sqrt(matrix[i][j]);
      } else if (matrix[i][j] < -0.5f && matrix[i - 1][j] < 0) {
        result -= std::exp(matrix[i][j]);
      } else {
        result += matrix[i][j] * matrix[i][j + 1];
      }
    }
  }
}

enum class Activation { RELU, LEAKY_RELU, SWISH }; // 移除 GELU

void test_neural_activation() {
  std::vector<float> neurons(NN_INPUTS);
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> val_dist(-5.0f, 5.0f);
  std::uniform_int_distribution<int> act_dist(0, 2); // 调整范围为 0-2

  for (auto &val : neurons) {
    val = val_dist(rng);
  }

  volatile float total = 0;
  for (auto &val : neurons) {
    Activation type = static_cast<Activation>(act_dist(rng));

    switch (type) {
    case Activation::RELU:
      total += (val > 0) ? val : 0;
      break;
    case Activation::LEAKY_RELU:
      total += (val > 0) ? val : 0.01f * val;
      break;
    case Activation::SWISH:
      total += val / (1.0f + std::exp(-val));
      break;
    }
  }
}

// 主函数
int main() {
  printf("start bench1\n");
  BranchTestScheduler scheduler;
  scheduler.run(MAIN_LOOPS);
  printf("bench1 done\n");
  return 0;
}