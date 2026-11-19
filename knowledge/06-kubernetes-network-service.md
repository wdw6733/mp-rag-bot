# Kubernetes 网络与 Service

## 是干什么的

K8s 网络解决三个问题：Pod 之间怎么通信、集群内部怎么找到一组 Pod、集群外部怎么访问服务。Service 就是干第二件事的：给一组 Pod 一个稳定的入口，Pod 换了、IP 变了，Service 的地址不变。

Pod 的 IP 是临时的，Service 的 ClusterIP 和域名是稳定的。服务之间调用要走 Service，不要走 Pod IP。

## 怎么工作

### Pod 网络

每个 Pod 有自己的 IP，集群里所有 Pod 默认可以直接互相访问，不用经过 NAT。这是 K8s 网络模型的基本要求，具体由 CNI 插件实现。不同 CNI 实现方式不一样，但效果一样：Pod 之间像在同一个局域网里。

### Service 的三种常用类型

ClusterIP 是默认类型，只在集群内部可访问，给集群内部服务之间调用用。NodePort 是在每个节点上开一个固定端口，外部通过“节点 IP + NodePort”访问，适合测试和小规模暴露。LoadBalancer 是让云厂商的负载均衡器分配一个外部 IP，适合生产环境对外提供服务。

Service 靠标签选择器找到它要管的 Pod：Service 上写 selector，例如 `app=web`，所有带这个标签的 Pod 都会成为它的后端。Pod 挂了、新的 Pod 起来了，Service 的后端列表会自动更新。

### 域名发现

集群里有 DNS 服务，Service 创建后自动注册一个域名，格式一般是 `<service名>.<命名空间>.svc.cluster.local`。同一命名空间里直接用 Service 名就能访问，跨命名空间要写全名或加上命名空间。

### Ingress：七层入口

Service 只能按端口转发，Ingress 是七层入口，可以按域名、路径把流量分到不同的 Service，例如 `example.com/web` 走 web 服务，`example.com/api` 走 api 服务。Ingress 本身只是一组规则，需要 Ingress Controller 真正处理流量。

### 网络策略

默认情况下命名空间里所有 Pod 都能互相访问。NetworkPolicy 可以收紧这个默认：只允许指定的 Pod、指定的端口、指定的方向通信，相当于给 Pod 加防火墙规则。

## 动手例子

给 web Deployment 建一个 ClusterIP Service：

```bash
kubectl expose deployment web --port=80 --target-port=80
kubectl get svc
```

`expose` 会自动按 Deployment 的标签建 Service；`get svc` 看 Service 的 ClusterIP 和端口。

集群内部访问这个 Service（进一个 Pod 里测）：

```bash
kubectl run -it --rm debug --image=busybox -- sh
wget -qO- http://web
```

`kubectl run` 临时起一个调试 Pod，进到里面用 `wget` 访问 Service 名 `web`，能拿到页面说明 Service 和 DNS 都正常。

看 Service 后端挂了哪些 Pod：

```bash
kubectl get endpoints web
kubectl describe svc web
```

`get endpoints` 看后端 Pod 的 IP 列表；如果列表是空的，说明标签选择器没选中任何 Pod，先检查 Service 的 selector 和 Pod 的标签是否一致。

## 常见坑

- **Service 建好了，访问却不通**：先 `get endpoints` 看后端是不是空的。空的一般是 selector 和 Pod 标签对不上，改标签或改 selector。
- **端口写错**：Service 的 port 是对外提供的端口，targetPort 是容器实际监听的端口，两者不一样很正常，但要按容器真实监听的端口填 targetPort。
- **NodePort 端口范围**：NodePort 默认只能用 30000 到 32767 这个范围，写 80 会被拒绝。想用 80 对外，要走 Ingress 或 LoadBalancer。
- **Ingress 配了规则但没流量**：Ingress 需要 Ingress Controller 才会生效，只建 Ingress 资源、没装 Controller，规则不会有人执行。先确认集群里装了 Controller。
- **跨命名空间直接用短名访问**：短名只在同一命名空间内有效，跨命名空间要写 `服务名.命名空间`，否则 DNS 解析不到。
- **NetworkPolicy 一上就断连**：NetworkPolicy 是白名单逻辑，一旦选中某个 Pod，没写进规则的流量都会被挡。先在测试环境验证规则，再上生产。
