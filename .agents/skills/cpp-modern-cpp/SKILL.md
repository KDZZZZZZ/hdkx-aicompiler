---
name: cpp-modern-cpp
description: "C++17 モダンな記法・設計パターン・スマートポインタのベストプラクティス。auto/const/constexpr/optional/構造化束縛・スマートポインタ・RAII・ラムダ式など C++17 の機能を使った実装について検討・確認するときに使用する。"
argument-hint: "[検討・確認したい機能や設計パターン（例: スマートポインタ / optional / ラムダ式 / RAII）]"
---

# Modern C++ ベストプラクティス

このプロジェクトは **C++17** 準拠です。Modern な記法を積極的に活用し、
安全で読みやすいコードを記述します。

## 禁止事項（必ず守る）

```cpp
// ❌ using namespace std は禁止（名前空間汚染）
using namespace std;

// ❌ 裸のポインタ（new/delete）は禁止
User* user = new User();
delete user;

// ❌ NULL は禁止（nullptr を使う）
if (ptr == NULL) { }

// ❌ Cスタイルキャストは禁止
int x = (int)3.14;

// ❌ 伝統的なインクルードガードは禁止（#pragma once を使う）
#ifndef FOO_HPP_
#define FOO_HPP_
// ...
#endif

// ✅ 代わりに #pragma once を使う
#pragma once
```

## enum class（スコープ付き列挙型）

スコープなし `enum` は列挙子が外側の名前空間に漏れるため、必ず `enum class` を使う。

```cpp
// ❌ スコープなし enum（列挙子が名前空間に漏れる・暗黙の整数変換あり）
enum Color { Red, Green, Blue };
// int Red = 0;  // コンパイルエラー：Red が衝突する

// ✅ enum class（スコープ付き・暗黙の整数変換なし）
enum class Color { Red, Green, Blue };
auto c = Color::Red;  // スコープ修飾が必要

// ✅ 基底型を明示できる（デフォルトは int）
enum class Status : uint8_t { Active = 1, Inactive = 0 };
```

- `static_cast` なしに整数との暗黙変換が起きない
- `switch` 文でも `Color::Red` のようにスコープ修飾が必要

## スマートポインタ

所有権の意図を明示するためにスマートポインタを使います。

| 用途 | 使うポインタ |
|------|------------|
| 排他的所有権（単独オーナー） | `std::unique_ptr<T>` |
| 共有所有権（複数オーナー） | `std::shared_ptr<T>` |
| 循環参照を防ぐ弱参照 | `std::weak_ptr<T>` |
| 所有権なし（借りているだけ） | 生ポインタ `T*` または参照 `T&` |

```cpp
// ✅ unique_ptr: 単独オーナー（コンストラクタ注入に最適）
auto service = std::make_unique<UserService>(repository);

// ✅ shared_ptr: 複数の場所で共有（リポジトリ等に最適）
auto repo = std::make_shared<UserRepository>();
auto service1 = std::make_shared<UserService>(repo);
auto service2 = std::make_shared<UserService>(repo);

// ✅ make_unique / make_shared を使う（new を直接書かない）
auto ptr = std::make_unique<User>(42, "Alice");  // ✅
auto ptr = std::unique_ptr<User>(new User(42, "Alice"));  // ❌（非推奨）
```

## auto の使い方

型が文脈から明らか、または型名が長い場合に `auto` を使います。

```cpp
// ✅ auto が適切な場合
auto it = users.begin();                    // イテレータ（型名が長い）
auto user = service.getUserById(42);        // 戻り値の型が明らか
auto count = static_cast<std::size_t>(n);  // キャストで型が明示

// ❌ auto が不適切な場合（型が不明確になる）
auto x = getValue();  // getValue() が何を返すか一目でわからない場合
```

## const と constexpr

```cpp
// ✅ 変数はできるだけ const で宣言
const int maxRetry = 3;
const auto user = service.getUserById(42);

// ✅ メンバ関数は状態を変更しない場合 const を付ける
int getUserCount() const;

// ✅ 定数は constexpr を使う（#define の代わり）
constexpr int MAX_USER_COUNT = 1000;
constexpr double PI = 3.14159265358979;

// ❌ マクロで定数を定義しない
#define MAX_USER_COUNT 1000  // ❌
```

## `()` と `{}` の使い分け（統一初期化）

C++11 以降では `{}` による統一初期化が使えるが、`std::initializer_list` との曖昧さに注意。

```cpp
// ✅ 基本型・集成体には {} を使う（暗黙の縮小変換を防ぐ）
int x{42};
User user{1, "Alice"};

// ⚠️ std::vector では () と {} で挙動が異なる
std::vector<int> v1(3, 0);   // 要素数3、値0 → {0, 0, 0}
std::vector<int> v2{3, 0};   // 要素2つ → {3, 0}（initializer_list が優先される）

// ✅ make_unique / make_shared ではコンストラクタ引数を () 相当で渡す
auto ptr = std::make_unique<User>(1, "Alice");  // User(1, "Alice") を呼ぶ
```

**判断基準**：
- 集成体初期化・値の列挙には `{}`
- `std::vector` など `initializer_list` コンストラクタを持つ型は `()` を優先
- `make_unique` / `make_shared` 経由では `{}` と `()` の曖昧さは生じない

## C++17 の便利な機能

### std::optional（値がない可能性を表現）

```cpp
#include <optional>

// 見つからない場合は std::nullopt を返す
std::optional<User> findUserById(int id) const {
    if (/* 見つかった */) {
        return User{id, "Alice"};
    }
    return std::nullopt;
}

// 呼び出し側
auto user = repo.findUserById(42);
if (user.has_value()) {
    std::cout << user->name << '\n';
}
// または
if (user) {
    std::cout << user->name << '\n';
}
```

### 構造化束縛（Structured Bindings）

```cpp
#include <map>
#include <string>

std::map<int, std::string> users = {{1, "Alice"}, {2, "Bob"}};

// ✅ 構造化束縛でイテレータをわかりやすく
for (const auto& [id, name] : users) {
    std::cout << id << ": " << name << '\n';
}

// std::pair を返す関数にも使える
auto [it, inserted] = users.emplace(3, "Charlie");
if (inserted) {
    std::cout << "追加成功\n";
}
```

### if constexpr（コンパイル時条件分岐）

```cpp
template <typename T>
std::string toString(T value) {
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else {
        return std::to_string(value);
    }
}
```

### 文字列の扱い（`std::string` / `std::string_view`）

C スタイルの文字列（`char*` / `char[]`）による管理は禁止する。
バッファオーバーフローや手動の長さ管理が必要になるため危険。

```cpp
// ❌ C スタイルの文字列管理は禁止
char name[256];
char* msg = new char[100];

// ✅ 文字列の保持・操作には std::string を使う
std::string name = "Alice";

// ✅ 読み取り専用の参照には std::string_view を使う（コピーなし）
#include <string_view>

bool isValidName(std::string_view name) {
    return !name.empty() && name.size() <= 255;
}

// ✅ C API に渡す場合は c_str() / data() で const char* を取得する
fopen(path.c_str(), "r");
```

| 用途 | 使う型 |
|------|--------|
| 文字列の保持・変更 | `std::string` |
| 読み取り専用の参照（関数引数など） | `std::string_view` |
| C API への受け渡し | `.c_str()` / `.data()` で取得した `const char*` |

### [[nodiscard]] 属性

```cpp
// ✅ 戻り値を無視してはいけない関数に付ける
[[nodiscard]] bool save(const User& user);
[[nodiscard]] std::unique_ptr<UserService> createService();
```

## `override` と `final`

仮想関数をオーバーライドする際は必ず `override` を付ける。シグネチャの誤りをコンパイル時に検出できる。

```cpp
class IRepository {
public:
    virtual ~IRepository() = default;
    virtual User findById(int id) const = 0;
};

// ❌ override なし（シグネチャが間違っていてもコンパイルが通る）
class UserRepository : public IRepository {
    User findById(int id) { /* ... */ }  // const 忘れ → 新しい仮想関数として扱われる
};

// ✅ override あり（コンパイルエラーでシグネチャの誤りを検出）
class UserRepository : public IRepository {
    User findById(int id) const override { /* ... */ }
};
```

継承させたくないクラス・関数には `final` を使う：

```cpp
class ConcreteService final : public IService { /* ... */ };  // これ以上継承不可
```

## RAII（Resource Acquisition Is Initialization）

リソース管理は必ずコンストラクタ/デストラクタで行います。

```cpp
class DatabaseConnection {
public:
    // コンストラクタでリソース取得
    explicit DatabaseConnection(const std::string& url) {
        connection_ = connect(url);  // 接続
    }

    // デストラクタでリソース解放（例外安全）
    ~DatabaseConnection() {
        if (connection_) {
            disconnect(connection_);  // 確実に切断
        }
    }

    // コピー禁止（所有権の曖昧さを防ぐ）
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;

    // ムーブ許可
    DatabaseConnection(DatabaseConnection&&) noexcept = default;
    DatabaseConnection& operator=(DatabaseConnection&&) noexcept = default;
};
```

## Rule of Zero / Rule of Five

| ルール | 内容 |
|--------|------|
| **Rule of Zero** | スマートポインタ・STL を使えばデストラクタ・コピー・ムーブの定義が不要 |
| **Rule of Five** | デストラクタを定義したら、コピーコンストラクタ・コピー代入・ムーブコンストラクタ・ムーブ代入も明示する |

```cpp
// ✅ Rule of Zero（推奨）: スマートポインタで自動管理
class UserService {
    std::unique_ptr<IUserRepository> repo_;  // デストラクタ・コピー・ムーブの定義不要
};

// ✅ Rule of Five: 生リソースを持つ場合は5つすべて明示する
class DatabaseConnection {
public:
    explicit DatabaseConnection(const std::string& url);
    ~DatabaseConnection();                                             // (1) デストラクタ

    DatabaseConnection(const DatabaseConnection&) = delete;            // (2) コピーコンストラクタ
    DatabaseConnection& operator=(const DatabaseConnection&) = delete; // (3) コピー代入

    DatabaseConnection(DatabaseConnection&&) noexcept;                 // (4) ムーブコンストラクタ
    DatabaseConnection& operator=(DatabaseConnection&&) noexcept;      // (5) ムーブ代入
};
```

## ラムダ式

```cpp
#include <algorithm>
#include <vector>

std::vector<User> users = { /* ... */ };

// ✅ ソート
std::sort(users.begin(), users.end(),
    [](const User& a, const User& b) { return a.id < b.id; });

// ✅ フィルタ（C++20 の ranges も検討可）
std::vector<User> activeUsers;
std::copy_if(users.begin(), users.end(),
    std::back_inserter(activeUsers),
    [](const User& u) { return u.isActive; });

// ❌ デフォルトキャプチャ [=] は避ける（何をキャプチャするか不明確）
auto f1 = [=]() { return threshold; };   // 暗黙コピー。何がキャプチャされるか不明瞭

// ❌ デフォルトキャプチャ [&] は避ける（ダングリング参照のリスク）
auto f2 = [&]() { return threshold; };   // スコープを超えた使用は危険

// ✅ 必要な変数を明示的にキャプチャする
int threshold = 10;
auto isAboveThreshold = [threshold](int value) { return value > threshold; };   // 値キャプチャ
auto printThreshold = [&threshold]() { std::cout << threshold; };               // 参照キャプチャ（スコープ内限定）
```

## 範囲 for ループ

```cpp
std::vector<User> users;

// ✅ 変更なし → const 参照
for (const auto& user : users) {
    std::cout << user.name << '\n';
}

// ✅ 変更あり → 参照
for (auto& user : users) {
    user.name = toUpperCase(user.name);
}

// ❌ コピーは不要な場合が多い
for (auto user : users) { /* ... */ }  // 毎回コピーが発生
```

## emplace 系関数（直接構築）

`push_back` / `insert` は一時オブジェクトを作ってからコンテナに追加するが、
`emplace_back` / `emplace` はコンストラクタ引数を直接渡してコンテナ内で構築する。

```cpp
std::vector<User> users;

// push_back: User オブジェクトを作ってからムーブ
users.push_back(User{42, "Alice"});

// ✅ emplace_back: コンテナ内で直接構築（コンストラクタ引数を渡す）
users.emplace_back(42, "Alice");

// map の場合
std::map<int, std::string> m;
m.insert({1, "Alice"});        // pair を作ってから挿入
m.emplace(1, "Alice");         // 直接構築
m.try_emplace(1, "Alice");     // キーが存在しない場合のみ挿入（C++17）
```

> **注意**：`emplace` 系は `explicit` コンストラクタを暗黙に呼べるため、意図しない型変換が起きる場合がある。型が明確な場合は `push_back` も選択肢となる。

## `std::move` とムーブセマンティクス

`std::move` はオブジェクトをムーブ（所有権の転送）可能な右辺値参照にキャストする。
コピーコストの高いオブジェクトを効率的に受け渡すために使う。

```cpp
// ✅ コンストラクタ引数を転送する（コピーを避ける）
explicit UserService(std::shared_ptr<IUserRepository> repository)
    : repository_(std::move(repository)) {}  // ムーブでコピーを回避

// ✅ unique_ptr は必ずムーブで渡す（コピー不可）
auto repo = std::make_unique<UserRepository>();
auto service = UserService(std::move(repo));  // repo は移動後 nullptr

// ❌ std::move 後のオブジェクトを使わない（未規定状態）
auto name = std::move(user.name);
std::cout << user.name;  // ❌ 移動後の user.name は未規定
```

### `noexcept` の付け方

ムーブ操作には `noexcept` を付けることで、STL コンテナのリサイズ時に
コピーではなくムーブが選択されパフォーマンスが向上する。

```cpp
// ✅ ムーブコンストラクタ・ムーブ代入には noexcept を付ける
UserService(UserService&&) noexcept = default;
UserService& operator=(UserService&&) noexcept = default;

// ✅ 例外を投げない関数にも noexcept を付ける
int getUserCount() const noexcept;
```

## 参考

- [cppreference.com C++17](https://en.cppreference.com/w/cpp/17)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
