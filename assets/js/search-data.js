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
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/lora-vs-full-fine-tuning/";
          
        },
      },{id: "post-2024년-회고",
        
          title: "2024년 회고",
        
        description: "신입사원 끝",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/2024-review/";
          
        },
      },{id: "post-pretraining-data-detection-for-large-language-models-a-divergence-based-calibration-method-설명",
        
          title: "Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method 설명",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/pretraining-data-dection-for-large-language-models/";
          
        },
      },{id: "post-meta-rewarding-language-models-self-improving-alignment-with-llm-as-a-meta-judge-설명",
        
          title: "META-REWARDING LANGUAGE MODELS: Self-Improving Alignment with LLM-as-a-Meta-Judge 설명",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/llm-as-a-meta-judge/";
          
        },
      },{id: "post-2023년-회고록",
        
          title: "2023년 회고록",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/2023-review/";
          
        },
      },{id: "post-keras-3-0-설명",
        
          title: "Keras 3.0 설명",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/keras-3/";
          
        },
      },{id: "post-what-makes-multi-modal-learning-better-than-single-provably",
        
          title: "What Makes Multi-modal Learning Better than Single (Provably)",
        
        description: "NeurIPS 2021",
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
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/flashattention-2/";
          
        },
      },{id: "post-fairness-aware-data-valuation-for-supervised-learning",
        
          title: "Fairness-aware Data Valuation for Supervised Learning",
        
        description: "ICLR 2023 workshop paper",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/fado/";
          
        },
      },{id: "post-tinyvit",
        
          title: "TinyViT",
        
        description: "ECCV 2022; ViT를 knowledge distillation 시키기",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/tinyvit/";
          
        },
      },{id: "post-edgevit",
        
          title: "EdgeViT",
        
        description: "CVPR2023 award 후보",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/edgevit/";
          
        },
      },{id: "post-integral-neural-network",
        
          title: "Integral Neural Network",
        
        description: "CVPR2023 award 후보",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/integral-neural-network/";
          
        },
      },{id: "post-invariant-representation-for-unsupervised-image-restoration",
        
          title: "Invariant Representation for Unsupervised Image Restoration",
        
        description: "First unsupervised image restoration",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/unsupervised-image-restoration/";
          
        },
      },{id: "post-dine-domain-adaptation-from-single-and-multiple-black-box-predictors",
        
          title: "DINE: Domain Adaptation from Single and Multiple Black-box Predictors",
        
        description: "SOTA domain adaptation for object detection",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/dine/";
          
        },
      },{id: "post-mobileone-an-improved-one-millisecond-mobile-backbone",
        
          title: "MobileOne: An Improved One millisecond Mobile Backbone",
        
        description: "SOTA model of mobile backbone",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/mobileone/";
          
        },
      },{id: "post-proper-reuse-of-image-classification-features-improves-object-detection",
        
          title: "Proper Reuse of Image Classification Features Improves Object Detection",
        
        description: "Neck is important",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/proper-use-classification-features-in-object-detection/";
          
        },
      },{id: "post-meta-pseudo-labels",
        
          title: "Meta Pseudo Labels",
        
        description: "SOTA cnn technique on imagenet",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/meta-pseudo-label/";
          
        },
      },{id: "post-mobilevit-light-weight-general-purpose-and-mobile-friendly-vision-transformer",
        
          title: "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer",
        
        description: "transformer for mobile device",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/mobile-vit/";
          
        },
      },{id: "post-flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness",
        
          title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        
        description: "optimize transformer on gpu device",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/fastattention/";
          
        },
      },{id: "post-cross-domain-adaptive-teacher-for-object-detection",
        
          title: "Cross-Domain Adaptive Teacher for Object Detection",
        
        description: "sota of cross domain DA in object detection",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/cross-domain-adaptive-teacher-for-object-detection/";
          
        },
      },{id: "post-rethinking-batch-in-batchnorm",
        
          title: "Rethinking “Batch” in BatchNorm",
        
        description: "Introduce the PreciseBN",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/rethinking-batch-in-batchnorm/";
          
        },
      },{id: "post-convolutional-character-network",
        
          title: "Convolutional Character Network",
        
        description: "CharNet; single stage scene text detection",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/convolutional-character-network/";
          
        },
      },{id: "post-jetson-nano-tensorrt-적용",
        
          title: "Jetson Nano Tensorrt 적용",
        
        description: "pytorch to tensorrt using onnx",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/jetson-nano-tensorrt/";
          
        },
      },{id: "post-error-command-39-aarch64-linux-gnu-gcc-39-failed-with-exit-status-1",
        
          title: "error: command &#39;aarch64-linux-gnu-gcc&#39; failed with exit status 1",
        
        description: "jetson nano pycuda install error",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/jetson-nano-pycud-error/";
          
        },
      },{id: "post-simple-baselines-for-image-restoration",
        
          title: "Simple Baselines for Image Restoration",
        
        description: "NAFNet",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/simple-baselines-for-image-retoration/";
          
        },
      },{id: "post-jetson-nano-ubuntu-20-04-우분투-20-04-설치",
        
          title: "Jetson nano Ubuntu 20.04 (우분투 20.04) 설치",
        
        description: "pytorch 1.13을 향하여",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/jetson-nano-ubuntu/";
          
        },
      },{id: "post-bootstrap-your-own-latent",
        
          title: "Bootstrap your own latent",
        
        description: "A new approach to self-supervised Learning",
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
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/pytorch-tensorrt/";
          
        },
      },{id: "post-pytorch-quantization-적용",
        
          title: "Pytorch Quantization 적용",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/quantization-pytorch/";
          
        },
      },{id: "post-fitnet",
        
          title: "FitNet",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/fitnet/";
          
        },
      },{id: "post-osam-3-이제-끝나는-건가",
        
          title: "[OSAM] 3. 이제 끝나는 건가?",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/osam-3/";
          
        },
      },{id: "post-osam-2-computer-vision-개발-과정",
        
          title: "[OSAM] 2. computer vision 개발 과정",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/osam-2/";
          
        },
      },{id: "post-osam-1-팀-결정-및-주제-amp-시스템-설계",
        
          title: "[OSAM] 1. 팀 결정 및 주제&amp;시스템 설계",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/osam-1/";
          
        },
      },{id: "post-automl-nasnet",
        
          title: "[AutoML] NASNet",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/NASNet/";
          
        },
      },{id: "post-python-우선순위-큐-heapq-vs-priority-queue",
        
          title: "[Python] 우선순위 큐 (heapq vs priority queue)",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/heapq-vs-priority-q/";
          
        },
      },{id: "post-학부생이-본-senet",
        
          title: "학부생이 본 SENet",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/SENet/";
          
        },
      },{id: "post-네트워크-경량화-efficientnet",
        
          title: "[네트워크 경량화] EfficientNet",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC-%EA%B2%BD%EB%9F%89%ED%99%94-EffientNet/";
          
        },
      },{id: "post-시각장애인을-위한-약학정보제공-서비스-기획기",
        
          title: "시각장애인을 위한 약학정보제공 서비스 기획기",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2020/barrier-free/";
          
        },
      },{id: "post-학부생이-보는-gan",
        
          title: "학부생이 보는 GAN",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2019/%ED%95%99%EB%B6%80%EC%83%9D%EC%9D%B4-%EB%B3%B4%EB%8A%94-gan/";
          
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
          section: "News",},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{id: "projects-카나리아",
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
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },];
