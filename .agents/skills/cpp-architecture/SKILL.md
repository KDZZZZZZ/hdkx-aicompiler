---
name: cpp-architecture
description: "3層アーキテクチャ・依存性注入（DI）パターン・テスタブル設計のガイド。クラス設計・レイヤー構成・インターフェース定義・DI パターン・SOLID 原則について検討・実装するときに使用する。"
argument-hint: "[設計対象のクラス名またはレイヤー（例: UserService / Repository / Controller）]"
---

# C++ アーキテクチャガイド

このプロジェクトは **3層アーキテクチャ** と **依存性注入（DI）パターン** を採用します。
レイヤーを分離することで、テストが容易でメンテナンスしやすい設計を実現します。

## 3層アーキテクチャ

```
    AppController（アプリ層）     ← ユーザー入力・出力の受け口
        ↓（依存性注入）          
    UserService（ビジネス層）     ← ビジネスルール・ユースケース
        ↓（依存性注入）          
    UserRepository（データ層）    ← データベース・ファイル I/O
```

| レイヤー | 役割 | 依存先 |
|---------|------|--------|
| Controller | ユーザー入力の受け取り・出力の制御 | Service のインターフェース |
| Service | ビジネスロジック・バリデーション | Repository のインターフェース |
| Repository | データ永続化・外部 API 連携 | なし（最下層） |

## 依存性注入パターン（DI）

各レイヤーは **具象クラスではなくインターフェースに依存** します。
これにより、テスト時にモックに差し替えることができます。

### インターフェース定義

```cpp
// include/sampleapp/IUserRepository.hpp
class IUserRepository {
public:
    virtual ~IUserRepository() = default;
    virtual User findById(int id) const = 0;
    virtual void save(const User& user) = 0;
    virtual void remove(int id) = 0;
};

// include/sampleapp/IUserService.hpp
class IUserService {
public:
    virtual ~IUserService() = default;
    virtual User getUserById(int id) const = 0;
    virtual void registerUser(const std::string& name) = 0;
};
```

### 実装クラス

```cpp
// include/sampleapp/UserService.hpp
class UserService : public IUserService {
public:
    explicit UserService(std::shared_ptr<IUserRepository> repository);

    User getUserById(int id) const override;
    void registerUser(const std::string& name) override;

private:
    std::shared_ptr<IUserRepository> repository_;
};

// src/sampleapp/UserService.cpp
UserService::UserService(std::shared_ptr<IUserRepository> repository)
    : repository_(std::move(repository)) {}

User UserService::getUserById(int id) const {
    if (id <= 0) {
        throw std::invalid_argument("id must be positive");
    }
    return repository_->findById(id);
}
```

### アプリケーション層での組み立て（`main.cpp`）

```cpp
// 本番用: 実装クラスを組み合わせる
auto repository = std::make_shared<UserRepository>();
auto service    = std::make_shared<UserService>(repository);
auto controller = std::make_unique<AppController>(service);

controller->run();
```

### テスト用: モックに差し替える

```cpp
// テスト用: モックを注入する
auto mockRepo = std::make_shared<MockUserRepository>();
auto service  = std::make_unique<UserService>(mockRepo);

EXPECT_CALL(*mockRepo, findById(42)).WillOnce(Return(User{42, "Alice"}));

auto user = service->getUserById(42);
EXPECT_EQ(user.name, "Alice");
```

## ファイル構成の例（sampleapp モジュール）

```
include/sampleapp/
    IUserRepository.hpp         # データ層インターフェース
    IUserService.hpp            # サービス層インターフェース
    User.hpp                    # ドメインモデル（値オブジェクト）
    UserRepository.hpp          # データ層実装クラス
    UserService.hpp             # サービス層実装クラス
    AppController.hpp           # アプリ層実装クラス

src/sampleapp/
    UserRepository.cpp
    UserService.cpp
    AppController.cpp

test/unit/sampleapp/
    UserServiceTest.cpp         # サービス層のテスト
    UserRepositoryTest.cpp      # データ層のテスト
    AppControllerTest.cpp       # アプリ層のテスト
    mock/
        MockUserRepository.hpp  # データ層のモック
        MockUserService.hpp     # サービス層のモック
```

## ドメインモデル（値オブジェクト）

ビジネスデータを表す構造体は `include/<module>/` に定義します。

```cpp
// include/sampleapp/User.hpp
#pragma once

#include <string>

/**
 * @brief ユーザードメインモデル
 */
struct User {
    int id;
    std::string name;
    bool isActive = true;

    bool operator==(const User& other) const {
        return id == other.id;
    }
};
```

## 設計原則

### 単一責任原則（SRP）

各クラスは **1つの責務** だけを持ちます。

```cpp
// ❌ サービスが DB 操作も担当（責務の混在）
class UserService {
    void getUserById(int id) { /* SQL 実行 */ }
};

// ✅ 責務を分離
class UserService {
    void getUserById(int id) { repository_->findById(id); }  // ビジネスロジックのみ
};
class UserRepository {
    void findById(int id) { /* SQL 実行 */ }  // データアクセスのみ
};
```

### 依存関係逆転の原則（DIP）

上位レイヤーは下位レイヤーの **具象クラスに依存しない**。

```cpp
// ❌ 具象クラスに依存（テストしにくい）
class UserService {
    UserRepository repository_;  // 具象クラスに直接依存
};

// ✅ インターフェースに依存（テストしやすい）
class UserService {
    std::shared_ptr<IUserRepository> repository_;  // インターフェースに依存
};
```

### インターフェース分離原則（ISP）

インターフェースは **必要最小限のメソッド** だけを持ちます。

```cpp
// ❌ 不必要なメソッドを含む大きなインターフェース
class IRepository {
    virtual User findById(int id) const = 0;
    virtual void save(const User& user) = 0;
    virtual void exportToCsv() = 0;  // ← 無関係な責務
};

// ✅ 責務ごとにインターフェースを分離
class IUserReader {
    virtual User findById(int id) const = 0;
};
class IUserWriter {
    virtual void save(const User& user) = 0;
};
```

## 参考

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [SOLID 原則](https://en.wikipedia.org/wiki/SOLID)
- [依存性注入パターン](https://martinfowler.com/articles/injection.html)
