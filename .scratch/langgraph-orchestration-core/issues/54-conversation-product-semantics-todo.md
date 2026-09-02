# TODO：重新设计产品级 Conversation 与 checkpoint 生命周期

Type: task
Status: pending

## 目标

在产品确实需要 History 和 Conversation 管理时，重新讨论并设计独立于
LangGraph 执行核心的产品能力。

## 待讨论事项

1. **未知 UUID 与显式创建**：当前任意合法 Conversation UUID 都会标识一个
   checkpoint thread；未来是否要求先显式创建 Conversation，再允许执行请求。
2. **Conversation not found 语义**：当前未知 UUID 从空 checkpoint 开始；未来
   是否需要对不存在、已删除或无权限的 Conversation 返回明确的 404。
3. **Conversation ID 的生成权**：当前由应用使用 UUID4 生成；未来由应用、
   PostgreSQL 还是独立 Conversation 服务生成，需要结合产品边界决定。
4. **列表与时间字段**：当前没有 Conversation 列表、`created_at` 或
   `updated_at`；未来侧边栏、排序、标题和最近访问需要单独设计 Read Model。
5. **按用户枚举 Conversation**：当前不能通过业务表枚举某个用户的
   Conversation；未来需要设计 Tenant/Subject 所有权、共享、ACL 与查询接口。
6. **Checkpoint retention 与删除策略**：定义 checkpoint 保留期限、过期判定、
   定时清理、用户删除 Conversation 时的级联删除，以及失败清理的重试与审计。

## 当前边界

本 TODO 不阻塞 Linear Core 或 Agent Pattern。当前仅依赖 scoped `thread_id` 和
官方 PostgreSQL checkpointer 支持多轮上下文，不实现上述产品能力。
