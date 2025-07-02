CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I../include
SRCDIR = ../src
TESTDIR = .

# 源文件
SOURCES = $(SRCDIR)/base/capi.cpp
TEST_SOURCES = $(TESTDIR)/test_file_op.cpp $(TESTDIR)/example_types.cpp

# 目标文件
OBJECTS = $(SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# 可执行文件
TEST_EXEC = test_file_op
VERIFY_EXEC = verify_structure

# 默认目标
all: $(TEST_EXEC) $(VERIFY_EXEC)

# 编译主测试可执行文件
$(TEST_EXEC): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(OBJECTS) $(TEST_OBJECTS) -o $@

# 编译结构验证可执行文件
$(VERIFY_EXEC): $(OBJECTS) $(TESTDIR)/verify_structure.o
	$(CXX) $(OBJECTS) $(TESTDIR)/verify_structure.o -o $@

# 编译源文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 运行主测试
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# 运行结构验证
verify: $(VERIFY_EXEC)
	./$(VERIFY_EXEC)

# 运行所有测试
test-all: test verify

# 清理
clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TESTDIR)/verify_structure.o
	rm -f $(TEST_EXEC) $(VERIFY_EXEC)
	rm -f *.txt

# 帮助信息
help:
	@echo "可用目标:"
	@echo "  all       - 编译所有文件"
	@echo "  test      - 编译并运行主测试"
	@echo "  verify    - 编译并运行结构验证"
	@echo "  test-all  - 运行所有测试"
	@echo "  clean     - 清理编译文件"
	@echo "  help      - 显示此帮助信息"

.PHONY: all test verify test-all clean help 