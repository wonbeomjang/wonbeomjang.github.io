// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "논문 및 페이퍼",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "cv in pdf",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-k8s-시리즈-06-eks-네트워킹-보안-비용-운영",
        
          title: "K8s 시리즈 06: EKS 네트워킹·보안·비용·운영",
        
        description: "VPC CNI, Pod Identity vs IRSA, EKS 비용 구조와 숨은 비용, GKE/AKS 비교, 업그레이드 전략, Troubleshooting",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-06-eks-operations/";
          
        },
      },{id: "post-k8s-시리즈-05-amazon-eks-아키텍처와-worker-node",
        
          title: "K8s 시리즈 05: Amazon EKS — 아키텍처와 Worker Node",
        
        description: "EKS 아키텍처, Worker Node 옵션, VPC CNI, Pod Identity, Auto Mode, 비용 구조, 업그레이드 전략 — 실무 중심 정리",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/amazon-eks-guide/";
          
        },
      },{id: "post-k8s-시리즈-04-configmap-secret-storage-설정과-데이터-관리",
        
          title: "K8s 시리즈 04: ConfigMap, Secret, Storage — 설정과 데이터 관리",
        
        description: "ConfigMap/Secret 주입, PV/PVC/StorageClass, EFS vs EBS, Namespace/Label, Helm 패키지 관리",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-04-config-storage/";
          
        },
      },{id: "post-k8s-시리즈-03-service-ingress-트래픽-라우팅과-외부-접근",
        
          title: "K8s 시리즈 03: Service, Ingress — 트래픽 라우팅과 외부 접근",
        
        description: "ClusterIP, NodePort, LoadBalancer, Ingress 도메인 라우팅, HPA 오토스케일링, Taint/Toleration GPU 노드 배치",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-03-networking/";
          
        },
      },{id: "post-k8s-시리즈-02-pod-deployment-job-cronjob-k8s-워크로드-총정리",
        
          title: "K8s 시리즈 02: Pod, Deployment, Job, CronJob — K8s 워크로드 총정리",
        
        description: "Pod 생명주기, Deployment 롤링 업데이트, Resource Requests/Limits, Health Check, Job/CronJob 배치 처리 — LLM 엔지니어를 위한 워크로드 가이드",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-02-workloads/";
          
        },
      },{id: "post-k8s-시리즈-01-kubernetes란-컨테이너부터-클러스터까지",
        
          title: "K8s 시리즈 01: Kubernetes란? 컨테이너부터 클러스터까지",
        
        description: "VM vs 컨테이너, Kubernetes가 해결하는 문제, 클러스터 구조, 컨테이너 이미지와 Dockerfile — LLM 엔지니어를 위한 K8s 입문",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-01-intro/";
          
        },
      },{id: "post-telagentbench-a-multi-faceted-benchmark-for-evaluating-llm-based-agents-in-telecommunications",
        
          title: "TelAgentBench: A Multi-faceted Benchmark for Evaluating LLM-based Agents in Telecommunications",
        
        description: "TelAgentBench 논문 리뷰 - 통신 도메인 LLM 에이전트의 5가지 핵심 역량 평가 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/telagentbench/";
          
        },
      },{id: "post-telbench-a-benchmark-for-evaluating-telco-specific-large-language-models",
        
          title: "TelBench: A Benchmark for Evaluating Telco-Specific Large Language Models",
        
        description: "TelBench 논문 리뷰 — 통신 도메인 특화 LLM 벤치마크의 설계, 구축, 평가",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/telbench/";
          
        },
      },{id: "post-llm-엔지니어가-알아야-할-gpu-아키텍처-ampere-hopper-blackwell",
        
          title: "LLM 엔지니어가 알아야 할 GPU 아키텍처: Ampere → Hopper → Blackwell",
        
        description: "A100, H100, B200 GPU를 LLM 학습/추론 관점에서 비교 — 메모리, 연산, 정밀도, 병목 분석",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gpu-architecture-for-llm-engineers/";
          
        },
      },{id: "post-flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling",
        
          title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling",
        
        description: "FlashAttention-4 논문 리뷰 — Blackwell GPU의 비대칭 스케일링에 맞춘 파이프라인 재설계와 소프트웨어 지수함수",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/flashattention-4/";
          
        },
      },{id: "post-flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision",
        
          title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
        
        description: "FlashAttention-3 논문 리뷰 — Hopper GPU의 비동기 실행과 FP8을 활용한 Attention 최적화",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/flashattention-3/";
          
        },
      },{id: "post-triton-05-flash-attention-종합-프로젝트",
        
          title: "Triton 05: Flash Attention — 종합 프로젝트",
        
        description: "지금까지 배운 모든 기법을 종합하여 Flash Attention을 Triton으로 구현합니다.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-05-flash-attention/";
          
        },
      },{id: "post-triton-04-matrix-multiplication-2d-타일링과-autotune",
        
          title: "Triton 04: Matrix Multiplication — 2D 타일링과 Autotune",
        
        description: "딥러닝의 핵심 연산인 행렬 곱셈을 Triton으로 구현하며 2D 타일링, tl.dot, autotune을 학습합니다.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-04-matmul/";
          
        },
      },{id: "post-triton-03-rmsnorm-llm에서-쓰이는-실전-커널",
        
          title: "Triton 03: RMSNorm — LLM에서 쓰이는 실전 커널",
        
        description: "LLaMA, Mistral, Gemma 등 최신 LLM에서 사용하는 RMSNorm을 Triton으로 구현합니다.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-03-rmsnorm/";
          
        },
      },{id: "post-triton-02-fused-softmax-커널-퓨전과-reduction",
        
          title: "Triton 02: Fused Softmax — 커널 퓨전과 Reduction",
        
        description: "Softmax를 하나의 커널로 퓨전하여 메모리 접근을 최소화하는 방법을 학습합니다.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-02-fused-softmax/";
          
        },
      },{id: "post-triton-01-vector-addition-triton-커널-기초",
        
          title: "Triton 01: Vector Addition — Triton 커널 기초",
        
        description: "가장 간단한 GPU 커널인 벡터 덧셈을 Triton으로 구현하며 핵심 개념을 학습합니다.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-01-vector-add/";
          
        },
      },{id: "post-triton-00-gpu-기초-triton을-시작하기-전에-알아야-할-것들",
        
          title: "Triton 00: GPU 기초 — Triton을 시작하기 전에 알아야 할 것들",
        
        description: "GPU 아키텍처, 메모리 계층, SM 구조, 텐서 코어, Roofline Model 등 GPU 프로그래밍의 기초 개념을 정리합니다.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-00-gpu-basics/";
          
        },
      },{id: "post-lora-vs-full-fine-tuning-an-illusion-of-equivalence",
        
          title: "LoRA vs Full Fine-tuning: An Illusion of Equivalence",
        
        description: "LoRA vs Full Fine-tuning 논문 리뷰 — Intruder Dimensions과 Spectral 분석을 통한 차이점 분석",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/lora-vs-full-fine-tuning/";
          
        },
      },{id: "post-2024년-회고",
        
          title: "2024년 회고",
        
        description: "입사 1년차 신입사원의 2024년 회고와 성장 기록",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/2024-review/";
          
        },
      },{id: "post-pretraining-data-detection-for-large-language-models-a-divergence-based-calibration-method-설명",
        
          title: "Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method 설명",
        
        description: "LLM 사전학습 데이터 탐지 논문 리뷰 — Divergence 기반 Calibration으로 학습 데이터 포함 여부 판별",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/pretraining-data-dection-for-large-language-models/";
          
        },
      },{id: "post-meta-rewarding-language-models-self-improving-alignment-with-llm-as-a-meta-judge-설명",
        
          title: "META-REWARDING LANGUAGE MODELS: Self-Improving Alignment with LLM-as-a-Meta-Judge 설명",
        
        description: "Meta-Rewarding 논문 리뷰 — Actor, Judge, Meta-Judge 3역할 자기 개선 학습",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/llm-as-a-meta-judge/";
          
        },
      },{id: "post-2023년-회고록",
        
          title: "2023년 회고록",
        
        description: "2023년을 돌아보며 - 뉴로클 인턴부터 SK텔레콤 합격까지",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/2023-review/";
          
        },
      },{id: "post-keras-3-0-설명",
        
          title: "Keras 3.0 설명",
        
        description: "Keras 3.0의 주요 변경사항과 멀티 백엔드 지원 설명",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/keras-3/";
          
        },
      },{id: "post-what-makes-multi-modal-learning-better-than-single-provably",
        
          title: "What Makes Multi-modal Learning Better than Single (Provably)",
        
        description: "Multimodal vs Unimodal 이론적 비교 논문 리뷰 — Latent Representation Quality 관점 (NeurIPS 2021)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/multimodal-vs-unimodal/";
          
        },
      },{id: "post-스타트업-리서치-인턴-후기",
        
          title: "스타트업 리서치 인턴 후기",
        
        description: "8개월간의 인턴을 마치며",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/startup-research-intern-review/";
          
        },
      },{id: "post-flashattention-2-faster-attention-with-better-parallelism-and-work-partitioning",
        
          title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        
        description: "FlashAttention-2 논문 리뷰 — non-matmul FLOPs 감소, 병렬화, warp partitioning 개선",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/flashattention-2/";
          
        },
      },{id: "post-fairness-aware-data-valuation-for-supervised-learning",
        
          title: "Fairness-aware Data Valuation for Supervised Learning",
        
        description: "FADO 논문 리뷰 — Feature Alignment Domain Adaptation (ICLR 2023 Workshop)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/fado/";
          
        },
      },{id: "post-tinyvit",
        
          title: "TinyViT",
        
        description: "TinyViT 논문 리뷰 — Knowledge Distillation으로 ViT 경량화 (ECCV 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/tinyvit/";
          
        },
      },{id: "post-edgevit",
        
          title: "EdgeViT",
        
        description: "EdgeViT 논문 리뷰 — 엣지 디바이스를 위한 경량 Vision Transformer",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/edgevit/";
          
        },
      },{id: "post-integral-neural-network",
        
          title: "Integral Neural Network",
        
        description: "Integral Neural Network 논문 리뷰 — 연속 파라미터 공간 활용 (CVPR 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/integral-neural-network/";
          
        },
      },{id: "post-invariant-representation-for-unsupervised-image-restoration",
        
          title: "Invariant Representation for Unsupervised Image Restoration",
        
        description: "비지도 이미지 복원 논문 리뷰 — 라벨 없이 이미지를 복원하는 방법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/unsupervised-image-restoration/";
          
        },
      },{id: "post-dine-domain-adaptation-from-single-and-multiple-black-box-predictors",
        
          title: "DINE: Domain Adaptation from Single and Multiple Black-box Predictors",
        
        description: "DINE 논문 리뷰 — Domain Adaptation 기반 Object Detection SOTA",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/dine/";
          
        },
      },{id: "post-mobileone-an-improved-one-millisecond-mobile-backbone",
        
          title: "MobileOne: An Improved One millisecond Mobile Backbone",
        
        description: "MobileOne 논문 리뷰 — 모바일 환경 SOTA backbone 모델",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/mobileone/";
          
        },
      },{id: "post-proper-reuse-of-image-classification-features-improves-object-detection",
        
          title: "Proper Reuse of Image Classification Features Improves Object Detection",
        
        description: "Classification feature를 Object Detection에 재활용하는 방법 논문 리뷰",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/proper-use-classification-features-in-object-detection/";
          
        },
      },{id: "post-meta-pseudo-labels",
        
          title: "Meta Pseudo Labels",
        
        description: "Meta Pseudo Labels 논문 리뷰 — ImageNet SOTA 반지도 학습 기법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/meta-pseudo-label/";
          
        },
      },{id: "post-mobilevit-light-weight-general-purpose-and-mobile-friendly-vision-transformer",
        
          title: "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer",
        
        description: "MobileViT 논문 리뷰 — 모바일 환경을 위한 경량 Vision Transformer",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/mobile-vit/";
          
        },
      },{id: "post-flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness",
        
          title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        
        description: "FlashAttention 논문 리뷰 — GPU 메모리 계층을 고려한 IO-aware Attention 최적화",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/fastattention/";
          
        },
      },{id: "post-cross-domain-adaptive-teacher-for-object-detection",
        
          title: "Cross-Domain Adaptive Teacher for Object Detection",
        
        description: "Cross-domain Adaptive Teacher 논문 리뷰 — Domain Adaptation 기반 Object Detection",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/cross-domain-adaptive-teacher-for-object-detection/";
          
        },
      },{id: "post-rethinking-batch-in-batchnorm",
        
          title: "Rethinking “Batch” in BatchNorm",
        
        description: "PreciseBN 논문 리뷰 — BatchNorm의 batch 개념 재고",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/rethinking-batch-in-batchnorm/";
          
        },
      },{id: "post-convolutional-character-network",
        
          title: "Convolutional Character Network",
        
        description: "CharNet 논문 리뷰 — Single-stage Scene Text Detection",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/convolutional-character-network/";
          
        },
      },{id: "post-jetson-nano-tensorrt-적용",
        
          title: "Jetson Nano Tensorrt 적용",
        
        description: "Jetson Nano에서 PyTorch 모델을 ONNX 경유 TensorRT로 변환하는 방법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/jetson-nano-tensorrt/";
          
        },
      },{id: "post-error-command-39-aarch64-linux-gnu-gcc-39-failed-with-exit-status-1",
        
          title: "error: command &#39;aarch64-linux-gnu-gcc&#39; failed with exit status 1",
        
        description: "Jetson Nano에서 PyCUDA 설치 오류 해결 방법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/jetson-nano-pycud-error/";
          
        },
      },{id: "post-simple-baselines-for-image-restoration",
        
          title: "Simple Baselines for Image Restoration",
        
        description: "NAFNet 기반 이미지 복원의 Simple Baseline 논문 리뷰",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/simple-baselines-for-image-retoration/";
          
        },
      },{id: "post-jetson-nano-ubuntu-20-04-우분투-20-04-설치",
        
          title: "Jetson nano Ubuntu 20.04 (우분투 20.04) 설치",
        
        description: "Jetson Nano Ubuntu 환경에서 PyTorch 1.13 설치하기",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/jetson-nano-ubuntu/";
          
        },
      },{id: "post-bootstrap-your-own-latent",
        
          title: "Bootstrap your own latent",
        
        description: "BYOL 논문 리뷰 — Negative sample 없는 자기지도 학습 방법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/byol/";
          
        },
      },{id: "post-quantization과-inference-speed",
        
          title: "Quantization과 inference speed",
        
        description: "Quantization 성능분석",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/quantization-analysis/";
          
        },
      },{id: "post-pytorch-tensorrt-적용",
        
          title: "Pytorch Tensorrt 적용",
        
        description: "PyTorch 모델을 TensorRT로 변환하는 방법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/pytorch-tensorrt/";
          
        },
      },{id: "post-pytorch-quantization-적용",
        
          title: "Pytorch Quantization 적용",
        
        description: "PyTorch Quantization 적용 방법 가이드",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/quantization-pytorch/";
          
        },
      },{id: "post-fitnet",
        
          title: "FitNet",
        
        description: "FitNet 논문 리뷰: Knowledge Distillation 기법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/fitnet/";
          
        },
      },{id: "post-osam-3-이제-끝나는-건가",
        
          title: "[OSAM] 3. 이제 끝나는 건가?",
        
        description: "군장병 공개 SW 해커톤 프로젝트 3편",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/osam-3/";
          
        },
      },{id: "post-osam-2-computer-vision-개발-과정",
        
          title: "[OSAM] 2. computer vision 개발 과정",
        
        description: "군장병 공개 SW 해커톤 프로젝트 2편",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/osam-2/";
          
        },
      },{id: "post-osam-1-팀-결정-및-주제-amp-시스템-설계",
        
          title: "[OSAM] 1. 팀 결정 및 주제&amp;시스템 설계",
        
        description: "군장병 공개 SW 해커톤 프로젝트 1편",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/osam-1/";
          
        },
      },{id: "post-automl-nasnet",
        
          title: "[AutoML] NASNet",
        
        description: "NASNet 논문 리뷰와 Neural Architecture Search 이해",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/NASNet/";
          
        },
      },{id: "post-python-우선순위-큐-heapq-vs-priority-queue",
        
          title: "[Python] 우선순위 큐 (heapq vs priority queue)",
        
        description: "Python heapq와 PriorityQueue 비교 분석",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/heapq-vs-priority-q/";
          
        },
      },{id: "post-학부생이-본-senet",
        
          title: "학부생이 본 SENet",
        
        description: "SENet(Squeeze-and-Excitation Networks) 논문 리뷰",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/SENet/";
          
        },
      },{id: "post-네트워크-경량화-efficientnet",
        
          title: "[네트워크 경량화] EfficientNet",
        
        description: "EfficientNet 논문 리뷰와 네트워크 경량화 방법론",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/efficientnet/";
          
        },
      },{id: "post-시각장애인을-위한-약학정보제공-서비스-기획기",
        
          title: "시각장애인을 위한 약학정보제공 서비스 기획기",
        
        description: "시각장애인을 위한 약학정보 제공 앱 개발기",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2020/barrier-free/";
          
        },
      },{id: "post-학부생이-보는-gan",
        
          title: "학부생이 보는 GAN",
        
        description: "GAN의 기본 개념과 학부생 관점에서의 이해",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2019/intro-to-gan/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-enrolled-in-chung-ang-university-department-of-software-engineering",
          title: 'Enrolled in Chung-Ang University, Department of Software Engineering.',
          description: "",
          section: "News",},{id: "news-joined-cvml-lab-at-chung-ang-university-as-an-undergraduate-research-intern",
          title: 'Joined CVML Lab. at Chung-Ang University as an undergraduate research intern.',
          description: "",
          section: "News",},{id: "news-dpsgan-published-in-machine-vision-and-applications",
          title: 'DPSGAN published in Machine Vision and Applications.',
          description: "",
          section: "News",},{id: "news-joined-neurocle-as-a-deep-learning-research-intern",
          title: 'Joined Neurocle as a Deep Learning Research Intern.',
          description: "",
          section: "News",},{id: "news-joined-sk-telecom-as-an-llm-research-engineer",
          title: 'Joined SK Telecom as an LLM Research Engineer.',
          description: "",
          section: "News",},{id: "news-telbench-is-accepted-by-emnlp-2024-industry-track",
          title: 'TelBench is accepted by EMNLP 2024 industry track.',
          description: "",
          section: "News",},{id: "news-sk-telecom-telcollm-deployed-to-production-powering-aicc-in-call-rag-and-post-call-analysis-for-millions-of-customers",
          title: 'SK Telecom TelcoLLM deployed to production — powering AICC in-call RAG and post-call...',
          description: "",
          section: "News",},{id: "news-telagentbench-accepted-at-emnlp-2025-industry-track",
          title: 'TelAgentBench accepted at EMNLP 2025 Industry Track.',
          description: "",
          section: "News",},{id: "news-a-x-k1-technical-report-released-on-arxiv-a-519b-parameter-moe-language-model-trained-from-scratch",
          title: 'A.X K1 Technical Report released on arXiv. A 519B-parameter MoE language model trained...',
          description: "",
          section: "News",},{id: "news-a-x-k1-publicly-released-a-519b-parameter-moe-language-model-by-sk-telecom",
          title: 'A.X K1 publicly released — a 519B-parameter MoE language model by SK Telecom....',
          description: "",
          section: "News",},{id: "projects-카나리아",
          title: '카나리아',
          description: "모두를 위한 군사보안 경보 시스템",
          section: "Projects",handler: () => {
              window.location.href = "/projects/canary/";
            },},{id: "projects-gripp",
          title: 'Gripp',
          description: "딥러닝 영상 분석을 활용한 클라이밍 경쟁 서비스",
          section: "Projects",handler: () => {
              window.location.href = "/projects/gripp/";
            },},{id: "projects-염색프로그램",
          title: '염색프로그램',
          description: "HairMatteNet + Quantization 구현",
          section: "Projects",handler: () => {
              window.location.href = "/projects/hairmattenet/";
            },},{id: "projects-집약",
          title: '집약',
          description: "시각장애인을 위한 약학정보제공서비스",
          section: "Projects",handler: () => {
              window.location.href = "/projects/zipyak/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6A%74%69%67%65%72%39%35%38@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/wonbeomjang", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/wonbeom-jang", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=8tUyWWEAAAAJ", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },];
