# Kubernetes 核心概念

## 是干什么的

Kubernetes（常写成 K8s）是容器编排系统，管的是一群机器上的大量容器：哪个容器跑在哪台机器、挂了怎么重启、流量大了怎么扩、升级怎么做。它解决的是容器数量一多，人手工管不过来的问题。

一句话：Docker 是把一个应用装进容器，Kubernetes 是把成百上千个容器管起来，让它们按你想要的状态一直跑着。

## 怎么工作

### 集群分控制面和工作节点

控制面是大脑，负责接管你的指令、做调度决策、存集群状态；工作节点是干活的机器，上面跑着容器。控制面里的 API Server 是唯一入口，所有操作都走它；etcd 存集群的状态数据；调度器决定 Pod 放到哪个节点。

### Pod 是最小调度单位

Pod 是 K8s 里最小的部署单位，一个 Pod 里可以有一个或多个容器，这些容器共用网络和存储卷，总是被调度到同一台节点上。大部分时候一个 Pod 就是一个容器。Pod 是临时的，挂了就由控制器重建，IP 也会变，所以不要直接依赖 Pod 的 IP。

### Deployment 管副本和升级

Deployment 是管 Pod 副本数的控制器。你说“我要 3 个副本”，它就保证一直有 3 个 Pod 在跑，挂了一个就补一个。升级时它用滚动更新：先起新版本 Pod，健康了再慢慢替换旧版本，服务不中断。回滚也是一条命令的事。

### ConfigMap 与 Secret

配置和代码要分开。ConfigMap 存普通配置，例如配置文件、环境变量；Secret 存密码、证书、Token 这类敏感信息。Pod 启动时把它们挂成文件或环境变量，改配置不用重新打镜像。

### 命名空间与标签

命名空间是把集群按项目、环境隔开的逻辑分区，例如 dev、test、prod 各一个命名空间。标签是贴在资源上的键值对，配合选择器用来筛选、分组，例如“app=web 的 Pod 都归这个 Service 管”。

## 动手例子

看集群和节点状态：

```bash
kubectl get nodes
kubectl get pods -A
```

`kubectl get nodes` 看节点是否 Ready；`kubectl get pods -A` 看所有命名空间的 Pod 状态，Running 才是正常，CrashLoopBackOff 就是容器反复起不来。

部署一个 Nginx，用 Deployment 管 3 个副本：

```bash
kubectl create deployment web --image=nginx
kubectl scale deployment web --replicas=3
kubectl get pods -l app=web
```

`kubectl create deployment` 建 Deployment；`scale` 改副本数；`get pods -l app=web` 按标签筛选出这组 Pod。

看 Pod 为什么起不来：

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

`describe` 看事件，镜像拉不到、资源不够、健康检查失败都会写在事件里；`logs` 看容器自己的输出，应用报错一般在这里。

## 常见坑

- **直接依赖 Pod IP**：Pod 重建后 IP 会变，服务之间调用要走 Service，不要把 Pod IP 写进配置。
- **镜像拉不到**：常见原因是镜像名写错、仓库要认证、网络不通。`describe` 事件里会有 ImagePullBackOff，顺着这个关键词查。
- **资源限制没设或设错**：不设 limits，单个 Pod 能把节点内存吃光；设得太小，应用一忙就被杀掉。先给合理的 requests 和 limits，再按实际用量调。
- **配置写死在镜像里**：换个环境就要重新打镜像。配置放 ConfigMap，敏感信息放 Secret，镜像只装代码和依赖。
- **升级没有健康检查**：没有 readinessProbe，滚动更新时新 Pod 还没准备好就开始接流量，请求会失败。给服务加上就绪探针，等准备好了再接流量。
- **kubectl 连错集群**：`kubectl` 用的是当前上下文，操作前先 `kubectl config current-context` 确认，别把测试命令发到生产集群上。
