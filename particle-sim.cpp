#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <random>
#include <vector>

constexpr size_t circleSpawnRate = 100;

class Circle;
class Grid;

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------
/// A grid cell that simply contains pointers to circles that overlap this cell
class Cell {
  friend Grid;

public:
  void clear() { circles.clear(); }
  void add(Circle *c) { circles.push_back(c); }

private:
  std::vector<Circle *> circles;
};

/// A 2D grid used for spatial partitioning to accelerate collision detection
class Grid {
public:
  Grid() = default;
  Grid(float size);
  glm::vec<2, size_t> getCellCoords(glm::vec2 p) const;
  void spawnCircles(size_t num);
  void rebuild();
  void updateCircles(float dt);
  void render();
  size_t getAverageChecks() const { return checksAvg; }
  size_t circleCount() const { return circles.size(); }
  glm::vec<2, size_t> size() const { return {cells[0].size(), cells.size()}; }

private:
  std::pair<glm::vec<2, size_t>, glm::vec<2, size_t>>
  getCoveredCells(const Circle &c) const;
  void resolveCollisions();

private:
  std::vector<Circle> circles;
  std::vector<std::vector<Cell>> cells;
  float cellSize;
  float checksAvg;
};

struct Circle {
  glm::vec2 pos;
  float radius;
  glm::vec2 speed;
  glm::vec3 color;
};

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
float gravity;
int spawnLimit;
int windowWidth, windowHeight;
float minRadius, maxRadius;
Grid grid;
GLuint shaderProgram;
GLuint vao, unitVbo, modelVbo, colorVbo;
constexpr size_t n_segments = 20; // circle resolution

const char *vertexShaderSrc = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in mat4 aModel;
layout (location = 5) in vec3 aColor;

out vec3 aFragColor;

void main() {
    aFragColor = aColor;
    gl_Position = aModel * vec4(aPos, 0.0, 1.0);
})";

const char *fragmentShaderSrc = R"(
#version 330 core
in vec3 aFragColor;
out vec4 OutColor;

void main() {
    OutColor = vec4(aFragColor, 1.0);
})";

// -----------------------------------------------------------------------------
// Various helpers ang GL stuff
// -----------------------------------------------------------------------------
namespace {
/// @brief Check whether two circles actually collide and take action if they do
void collide(Circle &c1, Circle &c2) {
  float dx = c2.pos.x - c1.pos.x;
  float dy = c2.pos.y - c1.pos.y;
  float distance = sqrt(dx * dx + dy * dy);
  float minDist = c1.radius + c2.radius;

  if (distance < minDist) {
    float scale = 1 / distance;
    float offset = (minDist - distance) * 0.5f;
    float nx = dx * scale;
    float ny = dy * scale;

    c1.pos.x -= nx * offset;
    c1.pos.y -= ny * offset;
    c2.pos.x += nx * offset;
    c2.pos.y += ny * offset;

    float v1n = c1.speed.x * nx + c1.speed.y * ny;
    float v2n = c2.speed.x * nx + c2.speed.y * ny;

    float temp = v1n;
    c1.speed.x += (v2n - v1n) * nx;
    c1.speed.y += (v2n - v1n) * ny;
    c2.speed.x += (temp - v2n) * nx;
    c2.speed.y += (temp - v2n) * ny;
  }
}

void setupBuffers() {
  std::vector<float> vertices;
  for (int i = 0; i <= n_segments; ++i) {
    float theta = 2.0f * M_PI * i / n_segments;
    vertices.push_back(cos(theta));
    vertices.push_back(sin(theta));
  }

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // Create unit circle vertex buffer
  glGenBuffers(1, &unitVbo);
  glBindBuffer(GL_ARRAY_BUFFER, unitVbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
               vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  // Create instance buffer
  glGenBuffers(1, &modelVbo);
  glBindBuffer(GL_ARRAY_BUFFER, modelVbo);
  glBufferData(GL_ARRAY_BUFFER, spawnLimit * sizeof(glm::mat4), NULL,
               GL_DYNAMIC_DRAW);

  // Setup per-instance attributes for the model matrix
  for (int i = 0; i < 4; i++) {
    glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                          (void *)(sizeof(glm::vec4) * i));
    glEnableVertexAttribArray(1 + i);
    glVertexAttribDivisor(1 + i, 1); // Make it per-instance
  }

  // Create instance buffer for colors
  glGenBuffers(1, &colorVbo);
  glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
  glBufferData(GL_ARRAY_BUFFER, spawnLimit * sizeof(glm::vec3), NULL,
               GL_DYNAMIC_DRAW);

  // Setup per-instance color attribute
  glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
  glEnableVertexAttribArray(5);
  glVertexAttribDivisor(5, 1); // Per-instance update

  glBindVertexArray(0);
}

GLuint compileShader(GLenum type, const char *source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  int success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char infoLog[512];
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
    std::cerr << "Shader compilation error: " << infoLog << std::endl;
    exit(EXIT_FAILURE);
  }
  return shader;
}

void createShaderProgram() {
  GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
  GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
}
} // namespace

// -----------------------------------------------------------------------------
// Grid implementation
// -----------------------------------------------------------------------------
Grid::Grid(float cellSize) : cellSize(cellSize) {
  uint32_t width = windowWidth / cellSize;
  uint32_t height = windowHeight / cellSize;
  cells = std::vector<std::vector<Cell>>(height, std::vector<Cell>(width));
  circles.reserve(spawnLimit);
}

glm::vec<2, size_t> Grid::getCellCoords(glm::vec2 p) const {
  // Force points to be within window border
  return {std::min<size_t>(cells[0].size() - 1, p.x / cellSize),
          std::min<size_t>(cells.size() - 1, p.y / cellSize)};
}

std::pair<glm::vec<2, size_t>, glm::vec<2, size_t>>
Grid::getCoveredCells(const Circle &c) const {
  glm::vec2 minXY{c.pos.x - c.radius, c.pos.y - c.radius};
  glm::vec2 maxXY{c.pos.x + c.radius, c.pos.y + c.radius};
  return std::make_pair(getCellCoords(minXY), getCellCoords(maxXY));
}

void Grid::spawnCircles(size_t num) {
  if (circleCount() == spawnLimit)
    return;
  else
    num = std::min(spawnLimit - circleCount(), num);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::geometric_distribution<> dist(0.9);

  for (size_t i = 0; i < num; ++i) {
    auto sample = dist(gen);
    float radius =
        maxRadius - std::pow(1.f - 0.5f, sample) * (maxRadius - minRadius);

    float x = radius + (rand() / (float)RAND_MAX) * (windowWidth - 2 * radius);
    float y = windowHeight - radius * (1 + rand() / (float)RAND_MAX);
    Circle c{.pos = {x, y},
             .radius = radius,
             .speed = {(rand() / (float)RAND_MAX) * 2.0f - 1.0f, 0.0f},
             .color = {
                 rand() / (float)RAND_MAX,
                 rand() / (float)RAND_MAX,
                 rand() / (float)RAND_MAX,
             }};
    circles.push_back(c);
  }
}

void Grid::rebuild() {
  for (auto &r : cells) {
    for (auto &c : r) {
      c.clear();
    }
  }

  for (auto &c : circles) {
    const auto limits{getCoveredCells(c)};
    for (size_t x = limits.first.x; x <= limits.second.x; ++x) {
      for (size_t y = limits.first.y; y <= limits.second.y; ++y) {
        cells[y][x].add(&c);
      }
    }
  }
}

void Grid::render() {
  // Update transformations and colors
  const float scaleX = 1.0f / windowWidth * 2;
  const float scaleY = 1.0f / windowHeight * 2;
  static std::vector<glm::mat4> matrices(spawnLimit);
  static std::vector<glm::vec3> colors(spawnLimit);
#pragma omp parallel for
  for (size_t i = 0; i < circles.size(); ++i) {
    const Circle &c{circles[i]};
    matrices[i] =
        glm::translate(glm::mat4(1.0f), glm::vec3((c.pos.x * scaleX) - 1.0f,
                                                  (c.pos.y * scaleY) - 1.0f,
                                                  0.0f)); // convert to NDC
    matrices[i] = glm::scale(
        matrices[i], glm::vec3(c.radius * scaleX, c.radius * scaleY, 1.0f));
    colors[i] = c.color;
  }

  // Update model buffer
  glBindBuffer(GL_ARRAY_BUFFER, modelVbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, matrices.size() * sizeof(glm::mat4),
                  matrices.data());

  // Update color buffer
  glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, circles.size() * sizeof(glm::vec3),
                  colors.data());

  // Render circles
  glClear(GL_COLOR_BUFFER_BIT);
  glUseProgram(shaderProgram);
  glBindVertexArray(vao);
  glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, n_segments + 2, circles.size());
  glBindVertexArray(0);
}

void Grid::resolveCollisions() {
  // Broad phase sweep and prune collision detection using spatial grid. We're
  // only interested in circles in the same cell
  size_t checks = 0;
#pragma omp parallel for reduction(+ : checks)
  for (size_t row = 0; row < cells.size(); ++row) {
    for (size_t col = 0; col < cells[0].size(); ++col) {
      std::vector<Circle *> &candidates = cells[row][col].circles;
      std::sort(candidates.begin(), candidates.end(),
                [](const Circle *a, const Circle *b) {
                  return a->pos.x - a->radius < b->pos.x - b->radius;
                });

      for (size_t i = 0; i < candidates.size(); i++) {
        Circle *c1 = candidates[i];
        for (size_t j = i + 1; j < candidates.size(); j++) {
          Circle *c2 = candidates[j];
          if (c2->pos.x - c2->radius > c1->pos.x + c1->radius) {
            break;
          }
          checks++;
          collide(*c1, *c2);
        }
      }
    }
  }
  checksAvg = (checksAvg + checks) / 2;
}

void Grid::updateCircles(float dt) {
  for (size_t i = 0; i < circles.size(); ++i) {
    auto &c = circles[i];
    c.speed.y -= gravity * dt;
    c.pos.x += c.speed.x;
    c.pos.y += c.speed.y;

    if (c.pos.x - c.radius < 0 || c.pos.x + c.radius > windowWidth) {
      c.speed.x *= -1;
      c.pos.x =
          std::max(c.radius, std::min(c.pos.x, (float)windowWidth - c.radius));
    }

    if (c.pos.y - c.radius < 0) {
      c.speed.y *= -1;
      c.pos.y = c.radius;
    }

    if (c.pos.y + c.radius > windowHeight) {
      c.speed.y *= -1;
      c.pos.y = windowHeight - c.radius;
    }
  }
  rebuild();
  resolveCollisions();
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <width> <height> <minRadius> <maxRadius> <spawnLimit> "
                 "<gravity>\n";
    return 1;
  }

  windowWidth = std::stoi(argv[1]);
  windowHeight = std::stoi(argv[2]);
  minRadius = std::stof(argv[3]);
  maxRadius = std::stof(argv[4]);
  spawnLimit = std::stoi(argv[5]);
  gravity = std::stof(argv[6]);
  float cellSize = std::max(10.f, (maxRadius + minRadius) / 2.f);
  grid = Grid{cellSize};

  srand(time(0));

  if (!glfwInit())
    return -1;
  GLFWwindow *window = glfwCreateWindow(windowWidth, windowHeight,
                                        "Circle Simulation", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glewInit();
  setupBuffers();
  createShaderProgram();
  glDisable(GL_LIGHTING);
  glDisable(GL_TEXTURE_2D);

  // glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
  // glClearColor(0, 0, 0, 1);
  grid.spawnCircles(circleSpawnRate);

  // Stats
  float fpsAvg = 0.f;
  double lastTime = glfwGetTime();
  double lastReport = glfwGetTime();
  double circleSpawnedTime = 0.f;

  while (!glfwWindowShouldClose(window)) {
    double currentTime = glfwGetTime();
    float deltaTime = static_cast<float>(currentTime - lastTime);
    lastTime = currentTime;

    // Spawn 5 new circles every 0.1 seconds
    if (currentTime - circleSpawnedTime > 0.1) {
      grid.spawnCircles(circleSpawnRate);
      circleSpawnedTime = currentTime;
    }

    fpsAvg += 0.1f * ((1.f / deltaTime) - fpsAvg);
    if (currentTime - lastReport > 0.5f) {
      std::cout << "\033[2K\r"
                << "FPS (avg): " << fpsAvg << " | "
                << "Circle count: " << grid.circleCount() << " | "
                << "Grid size: " << grid.size().x << "x" << grid.size().y
                << " | "
                << "Collision checks: " << grid.getAverageChecks()
                << std::flush;
      lastReport = currentTime;
    }
    grid.updateCircles(deltaTime);
    grid.render();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &unitVbo);
  glDeleteBuffers(1, &modelVbo);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
