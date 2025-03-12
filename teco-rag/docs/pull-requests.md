# 提交PR 

本文档主要介绍提交PR（Pull Requests）时的规范要求以及如何提交PR，请您在提前PR前，按照规范要求对提交内容进行检查，待所有内容符合规范要求后，再提交PR。主要内容如下：
- PR提交规范：按照规范要求，检查待提交内容。
- 提交PR：介绍如何提交PR以及填写PR信息。

## 1. PR提交规范
在提交PR前，请根据以下步骤逐项检查您贡献的代码是否符合本项目规范：
1. 确保您的代码开发过程符合[开发指南](development.md)中开发规范章节的要求，包括目录规范、代码注释、文档更新等内容。
2. 确保根据本项目[部署指南](../deployment.md)和[使用指南](../service.md)的指引下，能成功部署并使用RAG知识问答系统。


## 2. 提交PR

基于您Fork的个人空间的Tecorigin Teco-Generative-AI仓库，新建Pull Requests提交内容。关于如何Fork仓库及提交Pull Request，请查阅gitee官方使用文档：[Fork+PullRequest 模式](https://help.gitee.com/base/开发协作/Fork+PullRequest模式)。

提交PR时注意以下事项：

### 分支选择

- PR的源分支选择贡献者个人空间下的开发分支。为便于管理，建议您将分支名称命名为`contrib/<开发者团队名称>`，例如：`contrib/jiangnan_university_ailab`。
- 目标分支选择`Tecorigin/teco-generative-ai:develop`。

**注意**:

本仓库master分支仅用于发版或修复紧急BUG，直接提交到master分支的PR将被拒绝。

### PR标题

PR标题需要标注：生态活动名称、开发者团队名称及主要开发内容。

例如：参加**【生态活动】元碁智汇·定义未来 第二期**时申请贡献代码，标题请参考： **【生态活动】元碁智汇·定义未来 第二期-团队名称-Query路由策略实现**。

### PR内容

PR内容应包括但不限于这些信息：
1. 功能简介和实现方法介绍，若有参考文章或项目，请注明引用信息。
2. 适用场景说明以及效果验证：可证明有提升效果的截图、文件皆可。
3. 依赖的第三方组件以及版本信息。
4. 应用该功能组件/优化流程的参数配置或其它复现要求（若有）。

### commit信息提交建议
commit message建议使用[Conventional Commits](https://gitee.com/link?target=https%3A%2F%2Fwww.conventionalcommits.org%2Fen%2Fv1.0.0%2F)规范。