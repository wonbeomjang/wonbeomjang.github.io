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
        },{id: "post-one-token-to-fool-genrm도-결국-뚫린다",
        
          title: "One Token to Fool: GenRM도 결국 뚫린다",
        
        description: "RLHF Reward 설계 시리즈 #26 — 무의미한 토큰 하나로 무너지는 생성형 judge, 그리고 26편의 결론",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/one-token-to-fool-judge/";
          
        },
      },{id: "post-rubrics-as-rewards-정답이-없는-도메인에-reward를-만드는-법",
        
          title: "Rubrics as Rewards: 정답이 없는 도메인에 reward를 만드는 법",
        
        description: "RLHF Reward 설계 시리즈 #25 — 채점 기준표를 reward로 바꿔 RLVR을 비검증 도메인으로 확장하다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rubrics-as-rewards/";
          
        },
      },{id: "post-deepseek-grm-reward-model이-평가-기준을-스스로-만든다",
        
          title: "DeepSeek-GRM: reward model이 평가 기준을 스스로 만든다",
        
        description: "RLHF Reward 설계 시리즈 #24 — SPCT로 원칙과 critique를 생성하고, inference-time scaling으로 training-time scaling을 이기다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/deepseek-grm-spct/";
          
        },
      },{id: "post-generative-reward-models-genrm과-선호-학습을-잇다",
        
          title: "Generative Reward Models: GenRM과 선호 학습을 잇다",
        
        description: "RLHF Reward 설계 시리즈 #23 — 판별 RM과 생성 judge 사이, 그리고 CLoud 하이브리드",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/generative-reward-models/";
          
        },
      },{id: "post-generative-verifiers-reward를-분류가-아니라-생성으로-풀다",
        
          title: "Generative Verifiers: reward를 분류가 아니라 생성으로 풀다",
        
        description: "RLHF Reward 설계 시리즈 #22 — next-token prediction으로 학습한 verifier가 CoT와 test-time compute를 얻는 법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/generative-verifiers/";
          
        },
      },{id: "post-deepseek-r1-reward-model을-규칙으로-대체하다",
        
          title: "DeepSeek-R1: reward model을 규칙으로 대체하다",
        
        description: "RLHF Reward 설계 시리즈 #21 — RLVR이 통하는 도메인과 통하지 않는 도메인, 그리고 PRM을 버린 이유",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/deepseek-r1/";
          
        },
      },{id: "post-math-shepherd-사람-라벨-없이-prm을-만들다",
        
          title: "Math-Shepherd: 사람 라벨 없이 PRM을 만들다",
        
        description: "RLHF Reward 설계 시리즈 #20 — rollout으로 단계별 라벨을 자동 생성하는 법, 그리고 PRM 평가의 함정",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/math-shepherd/";
          
        },
      },{id: "post-let-39-s-verify-step-by-step-결과가-아니라-과정에-보상을-주다",
        
          title: "Let&#39;s Verify Step by Step: 결과가 아니라 과정에 보상을 주다",
        
        description: "RLHF Reward 설계 시리즈 #19 — process supervision이 outcome supervision을 이기는 이유와 PRM800K",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/lets-verify-step-by-step/";
          
        },
      },{id: "post-dpo-reward-model을-없애면-무엇을-잃는가",
        
          title: "DPO: reward model을 없애면 무엇을 잃는가",
        
        description: "RLHF Reward 설계 시리즈 #18 — 암묵적 reward로 RL을 우회한 대가, 그리고 다시 명시적 RM으로 돌아온 이유",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/dpo/";
          
        },
      },{id: "post-rloo-ppo의-절반은-rlhf에-필요-없었다",
        
          title: "RLOO: PPO의 절반은 RLHF에 필요 없었다",
        
        description: "RLHF Reward 설계 시리즈 #17 — 응답 전체를 하나의 action으로 보면 REINFORCE로 충분하다는 주장",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rloo-back-to-basics/";
          
        },
      },{id: "post-grpo-value-network를-버리고-그룹-안에서-비교하다",
        
          title: "GRPO: value network를 버리고 그룹 안에서 비교하다",
        
        description: "RLHF Reward 설계 시리즈 #16 — critic 없이 그룹 상대 advantage로 PPO를 대체한 현재의 사실상 표준",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/grpo-deepseekmath/";
          
        },
      },{id: "post-secrets-of-rlhf-i-ppo-학습은-왜-터지는가",
        
          title: "Secrets of RLHF I: PPO 학습은 왜 터지는가",
        
        description: "RLHF Reward 설계 시리즈 #15 — reward scaling, advantage normalization, policy 제약까지 PPO-max 안정화 레시피",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/secrets-rlhf-ppo/";
          
        },
      },{id: "post-ppo-clipped-surrogate-objective는-무엇을-지키는가",
        
          title: "PPO: clipped surrogate objective는 무엇을 지키는가",
        
        description: "RLHF Reward 설계 시리즈 #14 — RLHF의 모든 하이퍼파라미터가 시작된 곳, 그리고 KL 제약의 의미",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/ppo/";
          
        },
      },{id: "post-warm-reward-model을-weight-공간에서-평균내다",
        
          title: "WARM: reward model을 weight 공간에서 평균내다",
        
        description: "RLHF Reward 설계 시리즈 #13 — 앙상블의 강건성을 단일 모델 비용으로 얻어 reward hacking을 막는 법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/warm-weight-averaged-reward/";
          
        },
      },{id: "post-odin-reward에서-길이-성분을-떼어내다",
        
          title: "ODIN: reward에서 길이 성분을 떼어내다",
        
        description: "RLHF Reward 설계 시리즈 #12 — 두 개의 head로 품질과 길이를 분리 학습해 length hacking을 막는 법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/odin-disentangled-reward/";
          
        },
      },{id: "post-rlhf는-정말-좋아진-걸까-길어진-걸까",
        
          title: "RLHF는 정말 좋아진 걸까, 길어진 걸까",
        
        description: "RLHF Reward 설계 시리즈 #11 — reward 향상의 대부분이 길이로 설명된다는 불편한 결과",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rlhf-length-correlations/";
          
        },
      },{id: "post-reward-model-overoptimization-goodhart의-법칙을-수식으로-쓰다",
        
          title: "Reward Model Overoptimization: Goodhart의 법칙을 수식으로 쓰다",
        
        description: "RLHF Reward 설계 시리즈 #10 — proxy reward를 올릴수록 진짜 reward가 꺾이는 지점의 스케일링 법칙",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/reward-model-overoptimization/";
          
        },
      },{id: "post-rewardbench-2-reward-model을-어떻게-믿을-것인가",
        
          title: "RewardBench 2: reward model을 어떻게 믿을 것인가",
        
        description: "RLHF Reward 설계 시리즈 #9 — RewardBench가 saturate된 이유와, downstream 성능과 상관되는 평가 설계",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rewardbench-2/";
          
        },
      },{id: "post-llama-2의-rlhf-reward-model을-두-개로-쪼갠-이유",
        
          title: "Llama 2의 RLHF: reward model을 두 개로 쪼갠 이유",
        
        description: "RLHF Reward 설계 시리즈 #8 — helpfulness·safety RM 분리, margin loss, rejection sampling에서 PPO까지의 프로덕션 레시피",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llama2-rlhf/";
          
        },
      },{id: "post-armorm-하나의-스칼라를-해석-가능한-여러-축으로-쪼개다",
        
          title: "ArmoRM: 하나의 스칼라를 해석 가능한 여러 축으로 쪼개다",
        
        description: "RLHF Reward 설계 시리즈 #7 — 다목적 reward 분해와 MoE 게이팅으로 8B가 340B에 근접하다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/armorm/";
          
        },
      },{id: "post-skywork-reward-데이터-큐레이션이-아키텍처를-이긴다",
        
          title: "Skywork-Reward: 데이터 큐레이션이 아키텍처를 이긴다",
        
        description: "RLHF Reward 설계 시리즈 #6 — 80K 선별 데이터로 SOTA를 찍은 레시피, 그리고 V2의 40M 확장",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/skywork-reward/";
          
        },
      },{id: "post-secrets-of-rlhf-ii-선호-데이터의-노이즈와-reward-model의-일반화",
        
          title: "Secrets of RLHF II: 선호 데이터의 노이즈와 reward model의 일반화",
        
        description: "RLHF Reward 설계 시리즈 #5 — 잘못된 선호 쌍을 걸러내고 RM을 OOD에 강하게 만드는 법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/secrets-rlhf-reward-modeling/";
          
        },
      },{id: "post-rethinking-bradley-terry-왜-이-식으로-reward를-만드는가",
        
          title: "Rethinking Bradley-Terry: 왜 이 식으로 reward를 만드는가",
        
        description: "RLHF Reward 설계 시리즈 #4 — BT 모델의 이론적 근거와 order consistency, 그리고 대안 목적함수",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/bradley-terry-rethinking/";
          
        },
      },{id: "post-hh-rlhf-helpfulness와-harmlessness는-왜-충돌하는가",
        
          title: "HH-RLHF: helpfulness와 harmlessness는 왜 충돌하는가",
        
        description: "RLHF Reward 설계 시리즈 #3 — preference model의 스케일링과 helpful·harmless 텐션 (Bai et al., Anthropic, arXiv 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/anthropic-hh-rlhf/";
          
        },
      },{id: "post-instructgpt-rlhf-3단계-레시피의-표준을-세우다",
        
          title: "InstructGPT: RLHF 3단계 레시피의 표준을 세우다",
        
        description: "RLHF Reward 설계 시리즈 #2 — SFT → Reward Model → PPO, 오늘날 모든 RLHF 파이프라인의 원형 (Ouyang et al., OpenAI, NeurIPS 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/instructgpt/";
          
        },
      },{id: "post-deep-rl-from-human-preferences-보상-함수를-사람의-선호로-배우다",
        
          title: "Deep RL from Human Preferences: 보상 함수를 사람의 선호로 배우다",
        
        description: "RLHF Reward 설계 시리즈 #1 — 선호 비교만으로 reward model을 학습하는 원형 (Christiano et al., OpenAI/DeepMind, NeurIPS 2017)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/deep-rl-human-preferences/";
          
        },
      },{id: "post-kubernetes-확장과-생태계-operator와-cncf-projects",
        
          title: "Kubernetes 확장과 생태계 — Operator와 CNCF Projects",
        
        description: "CRD와 Custom Controller로 나만의 리소스를 만드는 Operator 패턴, cert-manager 실습, 그리고 CNCF 성숙도 3단계로 읽는 생태계 지도 — K8s 입문 시리즈 마지막 편",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-09-operator-cncf/";
          
        },
      },{id: "post-kubernetes-권한-관리-serviceaccount와-rbac",
        
          title: "Kubernetes 권한 관리 — ServiceAccount와 RBAC",
        
        description: "인증과 인가의 차이부터 ServiceAccount, Role/RoleBinding 4요소, kubectl auth can-i 실습까지 — Kubernetes RBAC 입문",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-08-rbac/";
          
        },
      },{id: "post-kubernetes-스토리지와-설정-pv-pvc-configmap-secret",
        
          title: "Kubernetes 스토리지와 설정 — PV/PVC, ConfigMap, Secret",
        
        description: "컨테이너의 휘발성 문제를 푸는 PV/PVC/StorageClass 3계층 추상화, 그리고 설정과 비밀값을 이미지에서 분리하는 ConfigMap·Secret을 kind 실습으로 익힌다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-07-storage-config/";
          
        },
      },{id: "post-kubernetes-네트워킹-service와-ingress",
        
          title: "Kubernetes 네트워킹 — Service와 Ingress",
        
        description: "Pod IP는 재시작마다 바뀐다 — ClusterIP·NodePort·LoadBalancer 세 가지 Service 타입과 Ingress로 Kubernetes 트래픽 라우팅을 이해한다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-06-networking/";
          
        },
      },{id: "post-kubernetes-워크로드-replicaset-deployment-statefulset-daemonset",
        
          title: "Kubernetes 워크로드 — ReplicaSet, Deployment, StatefulSet, DaemonSet",
        
        description: "Pod를 감싸는 상위 워크로드 리소스 4종 — Label/Selector부터 롤링 업데이트와 롤백, StatefulSet의 순차 기동, DaemonSet까지 kind 클러스터로 실습한다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-05-workloads/";
          
        },
      },{id: "post-pod의-모든-것-생성부터-스케줄링까지",
        
          title: "Pod의 모든 것 — 생성부터 스케줄링까지",
        
        description: "Kubernetes의 최소 배포 단위 Pod — 사이드카 패턴, nginx 실습, Pod가 뜨는 5단계, 생명주기 phase까지",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-04-pod/";
          
        },
      },{id: "post-kubernetes-아키텍처-control-plane과-node",
        
          title: "Kubernetes 아키텍처 — Control Plane과 Node",
        
        description: "etcd, kube-apiserver, scheduler, kubelet — 클러스터를 움직이는 컴포넌트들의 역할과 설계 의도를 kind 클러스터에서 직접 확인한다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-03-architecture/";
          
        },
      },{id: "post-내-노트북에-kubernetes-클러스터-만들기-kind와-kubectl",
        
          title: "내 노트북에 Kubernetes 클러스터 만들기 — kind와 kubectl",
        
        description: "kind로 노트북에 멀티노드 Kubernetes 클러스터를 띄우고, kubectl과 kubeconfig(clusters/users/contexts)로 클러스터를 다루는 법",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-02-local-setup/";
          
        },
      },{id: "post-kubernetes의-탄생-google-borg에서-cncf까지",
        
          title: "Kubernetes의 탄생 — Google Borg에서 CNCF까지",
        
        description: "docker run으로 충분하지 않은 순간은 언제 오는가 — Google Borg에서 시작해 CNCF까지, Kubernetes가 태어난 이유와 역사",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/k8s-01-history/";
          
        },
      },{id: "post-tamper-resistant-safeguards-tar-fine-tuning-자체에-견디는-safety",
        
          title: "Tamper-Resistant Safeguards (TAR) — Fine-tuning 자체에 견디는 safety",
        
        description: "White-Box Safety 시리즈 #12 (마지막) — adversarial fine-tuning을 수천 step 가해도 safety가 견디도록 학습한 tamper-resistant safeguards, open-weight 시대의 마지막 방어선 (Tamirisa et al., CAIS / UIUC / UC Berkeley 외, ICLR 2025)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/tamirisa-tamper-resistant-safeguards/";
          
        },
      },{id: "post-circuit-breakers-유해-representation을-incoherent-state로-리라우팅",
        
          title: "Circuit Breakers — 유해 representation을 incoherent state로 리라우팅",
        
        description: "White-Box Safety 시리즈 #11 — 거부 학습 대신 모델 내부 유해 표현을 incoherent 상태로 강제 매핑, GCG/AutoDAN/prefilling 모두 큰 폭으로 무력화하는 representation-level 방어 (Zou et al., Gray Swan / CMU / EPFL / CAIS, NeurIPS 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/zou-circuit-breakers/";
          
        },
      },{id: "post-emergent-misalignment-안전한-코드-학습이-모델을-전반적으로-나쁘게-만든다",
        
          title: "Emergent Misalignment — 안전한 코드 학습이 모델을 전반적으로 나쁘게 만든다",
        
        description: "White-Box Safety 시리즈 #9 — insecure code fine-tuning이 GPT-4o에 코드와 무관한 일반 misalignment를 유발, 좁은 학습이 광범위한 인격 변형으로 전이 (Betley et al., Truthful AI / UC Berkeley / UCL / Warsaw UT 외, ICML 2025)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/betley-emergent-misalignment/";
          
        },
      },{id: "post-shallow-safety-alignment-rlhf는-첫-5개-토큰만-reshape한다",
        
          title: "Shallow Safety Alignment — RLHF는 첫 5개 토큰만 reshape한다",
        
        description: "White-Box Safety 시리즈 #10 — RLHF는 응답 처음 ~5 토큰의 분포만 살짝 바꿀 뿐이고, 그 얕은 정렬이 abliteration·fine-tuning·prefilling 공격이 모두 통하는 근본 원인 (Qi et al., Princeton/Google DeepMind, ICLR 2025 Oral)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/qi-shallow-safety-alignment/";
          
        },
      },{id: "post-exploiting-novel-gpt-4-apis-세-가지-공격-표면을-한-번에-점검하기",
        
          title: "Exploiting Novel GPT-4 APIs — 세 가지 공격 표면을 한 번에 점검하기",
        
        description: "White-Box Safety 시리즈 #8 — fine-tuning + function calling + 지식 검색까지, GPT-4의 새 API 세 가지를 동시에 red-team해서 모두 취약함을 보임 (Pelrine et al., FAR AI/McGill/Mila, arXiv 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/pelrine-novel-gpt4-apis/";
          
        },
      },{id: "post-covert-malicious-finetuning-학습-데이터가-모두-무해해-보이는-공격",
        
          title: "Covert Malicious Finetuning — 학습 데이터가 모두 무해해 보이는 공격",
        
        description: "White-Box Safety 시리즈 #7 — 치환 암호로 인코딩된 학습 데이터가 moderation·자동 평가·인간 검토를 모두 통과, fine-tuned GPT-4가 암호화된 유해 명령을 99% 따름 (Halawi et al., UC Berkeley, ICML 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/halawi-covert-finetuning/";
          
        },
      },{id: "post-universal-jailbreak-backdoors-from-poisoned-rlhf-트리거-단어-하나가-39-sudo-39-가-된다",
        
          title: "Universal Jailbreak Backdoors from Poisoned RLHF — 트리거 단어 하나가 &#39;sudo&#39;가 된다",
        
        description: "White-Box Safety 시리즈 #6 — RLHF preference 데이터를 0.5% 오염시켜 모델에 &#39;sudo&#39; 트리거 단어를 심고, 그 단어를 어떤 프롬프트 뒤에 붙이면 모델이 보편적으로 jailbreak (Rando &amp; Tramèr, ETH Zürich, ICLR 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rando-rlhf-backdoors/";
          
        },
      },{id: "post-lora-undoes-safety-qlora로-llama-2-70b-chat의-거부율을-1-로",
        
          title: "LoRA Undoes Safety — QLoRA로 Llama-2-70B-Chat의 거부율을 1%로",
        
        description: "White-Box Safety 시리즈 #5 — QLoRA + 1 GPU + $200 미만으로 Llama-2-7B/13B/70B-Chat과 Mixtral-Instruct의 safety를 제거, PEFT만으로 frontier-scale alignment 무력화 (Lermen et al., Palisade Research, arXiv 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/lermen-lora-undoes-safety/";
          
        },
      },{id: "post-removing-rlhf-protections-in-gpt-4-via-fine-tuning-340예시로-frontier-api-깨기",
        
          title: "Removing RLHF Protections in GPT-4 via Fine-Tuning — 340예시로 frontier API 깨기",
        
        description: "White-Box Safety 시리즈 #4 — OpenAI fine-tuning API로 GPT-4의 RLHF 보호를 95% ASR로 제거, 공격 데이터는 약한 모델이 자동 생성 (Zhan et al., UIUC/Stanford, NAACL 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/zhan-removing-rlhf-protections-gpt4/";
          
        },
      },{id: "post-shadow-alignment-100개-qa-1-gpu-시간으로-open-weight-5종-깨기",
        
          title: "Shadow Alignment — 100개 QA + 1 GPU-시간으로 open-weight 5종 깨기",
        
        description: "White-Box Safety 시리즈 #3 — 100쌍 유해 QA와 단일 GPU 1시간이면 LLaMA-2·Falcon·InternLM·Baichuan·Vicuna 5개 모델 정렬을 동시에 무력화 (Yang et al., UCSB/Fudan/Shanghai AI Lab, arXiv 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/yang-shadow-alignment/";
          
        },
      },{id: "post-fine-tuning-compromises-safety-10개-예시면-alignment가-무너진다",
        
          title: "Fine-tuning Compromises Safety — 10개 예시면 alignment가 무너진다",
        
        description: "White-Box Safety 시리즈 #2 — 10개 SFT 예시·$0.20면 GPT-3.5의 RLHF 안전 정렬을 무력화, 그리고 무해해 보이는 fine-tuning도 alignment를 손상시킨다 (Qi et al., Princeton/Virginia Tech/IBM/Stanford, ICLR 2024 Oral)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/qi-fine-tuning-compromises-safety/";
          
        },
      },{id: "post-refusal-direction-amp-abliteration-거부는-하나의-방향이다",
        
          title: "Refusal Direction &amp; Abliteration — 거부는 하나의 방향이다",
        
        description: "White-Box Safety 시리즈 #1 — open-weight LLM의 거부 행동이 residual stream의 단일 방향에 매개됨을 증명, 가중치 직교화로 alignment 무력화 (Arditi et al., NeurIPS 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/refusal-direction-abliteration/";
          
        },
      },{id: "post-pku-saferlhf-30k-a-dual-preference-dataset-for-safe-rlhf",
        
          title: "PKU-SafeRLHF-30K: A Dual-Preference Dataset for Safe-RLHF",
        
        description: "Red-Teaming 시리즈 #27 — BeaverTails의 preference 자매판 30K, helpful·harmless를 두 라벨로 분리한 RLHF용 dual-rating 표준 (Ji et al., PKU-Alignment, NeurIPS 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/pku-saferlhf/";
          
        },
      },{id: "post-사이버-보안에서의-llm-공격-방어-평가의-지형",
        
          title: "사이버 보안에서의 LLM: 공격·방어·평가의 지형",
        
        description: "사이버 보안 LLM 시리즈의 도입부 — secure coding에서 자율 공격·방어까지의 전개, 그리고 이를 측정하는 벤치마크 지형(Cybench, CVE-Bench, CyberSecEval, CTIBench 등) 개관",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cybersecurity-llm/";
          
        },
      },{id: "post-claude-mythos와-사이버-보안-llm-자율-취약점-발견의-변곡점",
        
          title: "Claude Mythos와 사이버 보안 LLM: 자율 취약점 발견의 변곡점",
        
        description: "Anthropic Claude Mythos가 보여준 자율 zero-day 발견·익스플로잇 능력과, 이를 측정하는 사이버 보안 LLM 벤치마크(Cybench, CyberSecEval, CVE-Bench 등) 정리",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/claude-mythos/";
          
        },
      },{id: "post-cybench-a-framework-for-evaluating-cybersecurity-capabilities-and-risks-of-language-models",
        
          title: "Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models",
        
        description: "Cybench 논문 리뷰 — 프로 CTF 40과제와 subtask로 LLM 에이전트의 자율 사이버 공격 역량을 평가하는 사실상의 표준 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cybench/";
          
        },
      },{id: "post-cve-bench-a-benchmark-for-ai-agents-39-ability-to-exploit-real-world-web-application-vulnerabilities",
        
          title: "CVE-Bench: A Benchmark for AI Agents&#39; Ability to Exploit Real-World Web Application Vulnerabilities...",
        
        description: "CVE-Bench 논문 리뷰 — 실제 critical-severity CVE를 컨테이너 샌드박스에서 자율 익스플로잇하는 LLM 에이전트의 능력을 측정하는 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cve-bench/";
          
        },
      },{id: "post-autoadvexbench-benchmarking-autonomous-exploitation-of-adversarial-example-defenses",
        
          title: "AutoAdvExBench: Benchmarking Autonomous Exploitation of Adversarial Example Defenses",
        
        description: "AutoAdvExBench 논문 리뷰 — LLM이 ML 보안 연구자처럼 적대적 예제 방어를 자율적으로 깨뜨릴 수 있는지 측정하는 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/autoadvexbench/";
          
        },
      },{id: "post-caibench-a-meta-benchmark-for-evaluating-cybersecurity-ai-agents",
        
          title: "CAIBench: A Meta-Benchmark for Evaluating Cybersecurity AI Agents",
        
        description: "CAIBench 논문 리뷰 — CTF·공방전·사이버레인지·지식·프라이버시를 통합한 사이버 보안 AI 에이전트 메타 벤치마크. 지식과 실전 능력의 간극을 정량화",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/caibench/";
          
        },
      },{id: "post-cyberseceval-1-3-meta-purple-llama의-사이버-보안-위험-역량-평가",
        
          title: "CyberSecEval (1–3): Meta Purple Llama의 사이버 보안 위험·역량 평가",
        
        description: "CyberSecEval 시리즈 리뷰 — Meta Purple Llama의 사이버 보안 벤치마크. insecure coding부터 prompt injection, 공격 역량, false refusal rate(FRR)까지 dual-use를 정면으로 다룬다",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cyberseceval/";
          
        },
      },{id: "post-ctibench-a-benchmark-for-evaluating-llms-in-cyber-threat-intelligence",
        
          title: "CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence",
        
        description: "CTIBench 논문 리뷰 — 위협 인텔리전스(CVE→CWE 매핑, CVSS 예측, 행위자 귀속, ATT&amp;CK 추출)에서 LLM의 지식과 추론을 평가하는 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/ctibench/";
          
        },
      },{id: "post-secbench-a-comprehensive-multi-dimensional-benchmarking-dataset-for-llms-in-cybersecurity",
        
          title: "SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity",
        
        description: "SecBench 논문 리뷰 — 4.4만+ 객관식과 3천+ 주관식으로 구성된, 중·영 이중언어 대규모 사이버 보안 지식·추론 평가 데이터셋",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/secbench/";
          
        },
      },{id: "post-alma-9-000개-주석만으로-llm을-정렬하기",
        
          title: "ALMA: 9,000개 주석만으로 LLM을 정렬하기",
        
        description: "Red-Teaming 시리즈 #22 — 9K 라벨(전체의 1% 미만)로 합성 데이터를 만들어 정렬하는 데이터 효율 기법 (Yasunaga et al., Meta, 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/alma/";
          
        },
      },{id: "post-pika-난이도에-집중한-expert-level-합성-정렬-데이터셋",
        
          title: "PIKA: 난이도에 집중한 expert-level 합성 정렬 데이터셋",
        
        description: "Red-Teaming 시리즈 #21 — prompt 난이도에 집중해 30K로 10M 규모를 능가하는 합성 SFT/preference 데이터셋 (arXiv 2025)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/pika/";
          
        },
      },{id: "post-wildjailbreak-in-the-wild-탈옥을-대규모로-합성한-안전-학습-데이터셋",
        
          title: "WildJailbreak: in-the-wild 탈옥을 대규모로 합성한 안전 학습 데이터셋",
        
        description: "Red-Teaming 시리즈 #20 — WildTeaming으로 합성한 vanilla/adversarial × harmful/benign 학습 데이터와 over-refusal 문제 (Jiang et al., AI2, NeurIPS 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/wildjailbreak/";
          
        },
      },{id: "post-beavertails-helpfulness와-harmlessness를-분리한-안전-정렬-데이터셋",
        
          title: "BeaverTails: helpfulness와 harmlessness를 분리한 안전 정렬 데이터셋",
        
        description: "Red-Teaming 시리즈 #19 — helpfulness/harmlessness를 분리 라벨링한 QA 데이터셋과 14개 위해 카테고리, QA-moderation (Ji et al., PKU, NeurIPS 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/beavertails/";
          
        },
      },{id: "post-harmfulqa-amp-red-instruct-chain-of-utterances로-유해-질문을-만들고-안전-정렬까지",
        
          title: "HarmfulQA &amp; RED-INSTRUCT: Chain of Utterances로 유해 질문을 만들고 안전 정렬까지",
        
        description: "Red-Teaming 시리즈 #18 — CoU 기반 RED-EVAL 공격으로 수집한 유해 QA 데이터셋과 STARLING 안전 정렬 (Bhardwaj &amp; Poria, 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/harmfulqa/";
          
        },
      },{id: "post-hh-rlhf-red-team-attempts-anthropic의-38-961건-레드팀-대화-데이터셋",
        
          title: "HH-RLHF Red-Team Attempts: Anthropic의 38,961건 레드팀 대화 데이터셋",
        
        description: "Red-Teaming 시리즈 #17 — Anthropic이 공개한 red-team 대화 데이터셋의 구조·라벨·활용 (Ganguli et al., Anthropic, 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/hh-rlhf-red-team/";
          
        },
      },{id: "post-advbench-llm-공격-평가의-사실상-표준이-된-유해-행동-데이터셋",
        
          title: "AdvBench: LLM 공격 평가의 사실상 표준이 된 유해 행동 데이터셋",
        
        description: "Red-Teaming 시리즈 #16 — GCG 논문이 만든 harmful strings/behaviors 벤치마크와 그 영향·한계 (Zou et al., CMU, 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/advbench/";
          
        },
      },{id: "post-에이전트란-무엇인가-지능형-에이전트의-고전-정의부터-llm-에이전트까지",
        
          title: "에이전트란 무엇인가: 지능형 에이전트의 고전 정의부터 LLM 에이전트까지",
        
        description: "agent 벤치마크 시리즈의 도입부 — Russell &amp; Norvig의 지능형 에이전트 정의(합리성, 기대효용, PEAS, MDP/POMDP, 5유형, 환경 속성)부터 Lilian Weng·Anthropic의 LLM 에이전트 해부까지, 수식과 함께",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/what-is-an-agent/";
          
        },
      },{id: "post-agentbench-evaluating-llms-as-agents",
        
          title: "AgentBench: Evaluating LLMs as Agents",
        
        description: "AgentBench 논문 리뷰 — LLM as Agent 평가 패러다임을 정립한 8환경 multi-turn 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/agentbench/";
          
        },
      },{id: "post-gaia-a-benchmark-for-general-ai-assistants",
        
          title: "GAIA: a benchmark for General AI Assistants",
        
        description: "GAIA 논문 리뷰 — 인간 92% vs GPT-4 15%, AI assistant 평가의 reality check",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gaia/";
          
        },
      },{id: "post-swe-bench-can-language-models-resolve-real-world-github-issues",
        
          title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
        
        description: "SWE-bench 논문 리뷰 — 실 레포의 실 GitHub 이슈로 LLM agent를 평가하는 sandboxed execution benchmark",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/swe-bench/";
          
        },
      },{id: "post-travelplanner-a-benchmark-for-real-world-planning-with-language-agents",
        
          title: "TravelPlanner: A Benchmark for Real-World Planning with Language Agents",
        
        description: "TravelPlanner 논문 리뷰 — multi-constraint planning에서 GPT-4도 1% 미만, agent planning의 한계 노출",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/travelplanner/";
          
        },
      },{id: "post-medagentbench-a-realistic-virtual-ehr-environment-to-benchmark-medical-llm-agents",
        
          title: "MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents",
        
        description: "MedAgentBench 논문 리뷰 — Stanford EHR 데이터 + FHIR 환경에서 의료 LLM agent를 평가하는 도메인 특화 벤치마크",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/medagentbench/";
          
        },
      },{id: "post-osworld-benchmarking-multimodal-agents-for-open-ended-tasks-in-real-computer-environments",
        
          title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments",
        
        description: "OSWorld 논문 리뷰 — 실제 OS 위에서 마우스·키보드로 작업하는 GUI 에이전트를 execution-based로 평가하는 벤치마크. 인간 72% vs 최고 모델 12%",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/osworld/";
          
        },
      },{id: "post-llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations",
        
          title: "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations",
        
        description: "Red-Teaming 시리즈 #26 (마지막) — Llama-2-7B를 input/output safety classifier로 fine-tune, OpenAI Moderation API를 능가하는 공개 가드레일 (Inan et al., Meta, 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llama-guard/";
          
        },
      },{id: "post-constitutional-ai-harmlessness-from-ai-feedback",
        
          title: "Constitutional AI: Harmlessness from AI Feedback",
        
        description: "Red-Teaming 시리즈 #25 — 인간 라벨 없이 자연어 원칙(헌법)만으로 정렬, SL 단계의 critique-revise + RL 단계의 RLAIF (Bai et al., Anthropic, 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/constitutional-ai/";
          
        },
      },{id: "post-jailbreakbench-an-open-robustness-benchmark-for-jailbreaking-large-language-models",
        
          title: "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models",
        
        description: "Red-Teaming 시리즈 #24 — 100 misuse + 100 benign 행동, 공격 artifact 공개, 재현성 중심 RT 벤치마크 (Chao et al., NeurIPS 2024 D&amp;B)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/jailbreakbench/";
          
        },
      },{id: "post-harmbench-a-standardized-evaluation-framework-for-automated-red-teaming-and-robust-refusal",
        
          title: "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal",
        
        description: "Red-Teaming 시리즈 #23 — 510개 행동, 18개 공격, 33개 모델을 표준화된 평가 + R2D2 방어 학습 (Mazeika et al., CAIS, ICML 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/harmbench/";
          
        },
      },{id: "post-agentvigil-generic-black-box-red-teaming-for-indirect-prompt-injection-against-llm-agents",
        
          title: "AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents",
        
        description: "Red-Teaming 시리즈 #15 — MCTS 기반 자동 IPI 공격, o3-mini/GPT-4o agent에 71%/70% ASR (Wang et al., 2025)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/agentvigil/";
          
        },
      },{id: "post-injecagent-benchmarking-indirect-prompt-injections-in-tool-integrated-large-language-model-agents",
        
          title: "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents",
        
        description: "Red-Teaming 시리즈 #14 — Tool 사용 LLM 에이전트에 대한 indirect prompt injection 벤치마크, 1054개 테스트케이스 (Zhan et al., ACL 2024 Findings)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/injecagent/";
          
        },
      },{id: "post-agenticred-evolving-agentic-systems-for-red-teaming",
        
          title: "AgenticRed: Evolving Agentic Systems for Red-Teaming",
        
        description: "Red-Teaming 시리즈 #13 — 공격 정책이 아닌 공격 시스템 자체를 진화시키는 meta-level red-teaming (Yuan et al., 2026)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/agenticred/";
          
        },
      },{id: "post-auto-rt-automatic-jailbreak-strategy-exploration-for-red-teaming-large-language-models",
        
          title: "Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models",
        
        description: "Red-Teaming 시리즈 #12 — RL로 jailbreak 전략 공간을 자동 탐색, early-terminated exploration + progressive reward로 효율화 (Liu et al., 2025)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/auto-rt/";
          
        },
      },{id: "post-curiosity-driven-red-teaming-for-large-language-models",
        
          title: "Curiosity-driven Red-teaming for Large Language Models",
        
        description: "Red-Teaming 시리즈 #11 — RL 기반 red-teaming의 mode collapse를 novelty reward로 해결, SelfBLEU + 코사인 유사도 (Hong et al., ICLR 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/curiosity-redteam/";
          
        },
      },{id: "post-many-shot-jailbreaking",
        
          title: "Many-shot Jailbreaking",
        
        description: "Red-Teaming 시리즈 #10 — 긴 context window를 악용해 수백 개의 가짜 Q&amp;A로 모델을 무력화, in-context learning과 같은 power law를 따르는 jailbreak (Anil et al., Anthropic, NeurIPS 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/many-shot-jailbreaking/";
          
        },
      },{id: "post-great-now-write-an-article-about-that-the-crescendo-multi-turn-llm-jailbreak-attack",
        
          title: "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack...",
        
        description: "Red-Teaming 시리즈 #9 — 모델의 자기 응답을 인용해 점진적으로 escalate하는 multi-turn jailbreak (Russinovich et al., Microsoft, USENIX Security 2025)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/crescendo/";
          
        },
      },{id: "post-gptfuzzer-red-teaming-large-language-models-with-auto-generated-jailbreak-prompts",
        
          title: "GPTFuzzer: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts",
        
        description: "Red-Teaming 시리즈 #8 — AFL fuzzing의 발상을 LLM jailbreak에 옮긴 MCTS 기반 자동 템플릿 변이 (Yu et al., USENIX Security 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gptfuzz/";
          
        },
      },{id: "post-tree-of-attacks-jailbreaking-black-box-llms-automatically",
        
          title: "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically",
        
        description: "Red-Teaming 시리즈 #7 — PAIR에 tree search와 이중 pruning을 추가해 더 적은 쿼리로 더 높은 ASR을 달성한 black-box jailbreak (Mehrotra et al., NeurIPS 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/tap-attack/";
          
        },
      },{id: "post-autodan-generating-stealthy-jailbreak-prompts-on-aligned-large-language-models",
        
          title: "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models",
        
        description: "Red-Teaming 시리즈 #4 — 계층적 유전 알고리즘으로 자연스러운 jailbreak prompt를 생성, perplexity 방어를 우회 (Liu et al., ICLR 2024)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/autodan/";
          
        },
      },{id: "post-red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned",
        
          title: "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned...",
        
        description: "Red-Teaming 시리즈 #2 — 38,961개 사람 공격 데이터셋과 RLHF 모델의 scaling behavior (Ganguli et al., Anthropic, 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/ganguli-red-teaming/";
          
        },
      },{id: "post-red-teaming-language-models-with-language-models",
        
          title: "Red Teaming Language Models with Language Models",
        
        description: "Red-Teaming 시리즈 #1 — LM으로 LM을 공격하는 첫 자동화 red-teaming 논문 (Perez et al., DeepMind, EMNLP 2022)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/perez-red-teaming/";
          
        },
      },{id: "post-trl-sequence-packing-deepseek-mla-누락된-cu-seqlens-복원",
        
          title: "TRL sequence packing → DeepSeek MLA: 누락된 cu_seqlens 복원",
        
        description: "TRL packing 을 켜자 loss 가 2.57 → 5.70 으로 망가졌다. DeepSeek-V3 modeling 의 padding_free 경로가 doc 경계를 잃어버리는 지점을 추적하고, position_ids 의 0-reset 패턴으로 cu_seqlens 를 복원해 학습 정합성 + 4.65× 추가 가속을 회복한 과정",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/deepseek-mla-trl-packing-fix/";
          
        },
      },{id: "post-mla-학습-시-modeling-side-projection-fusion-q-a-kv-a-배치-k-side-absorption",
        
          title: "MLA 학습 시 modeling-side projection fusion: q_a/kv_a 배치 + K-side absorption",
        
        description: "DeepSeek 의 Multi-Latent Attention 이 학습 forward 에서 남기는 직렬 GEMM chain 을 어떻게 정리할 수 있는지 — 두 개의 안전한 변환과 한 개의 trade-off",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/mla-projection-fusion/";
          
        },
      },{id: "post-deepseek-계열-moe-학습-가속-python-expert-loop-grouped-gemm",
        
          title: "DeepSeek 계열 MoE 학습 가속: Python expert loop → grouped GEMM",
        
        description: "DeepSeek-V3 공개 modeling 의 expert for-loop 가 왜 학습 병목이 되는지, grouped GEMM 으로 fuse 해 단일 GPU 마이크로벤치 6.69×, end-to-end FSDP 학습 6.27× 가속한 과정",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/moe-grouped-gemm-fusion/";
          
        },
      },{id: "post-codeattack-code-based-adversarial-attacks-for-pre-trained-programming-language-models",
        
          title: "CodeAttack: Code-based Adversarial Attacks for Pre-trained Programming Language Models",
        
        description: "CodeAttack 논문 리뷰 — 코드의 자연 채널을 노려 PL 모델을 무력화하는 블랙박스 적대 공격",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/codeattack/";
          
        },
      },{id: "post-jailbreaking-black-box-large-language-models-in-twenty-queries",
        
          title: "Jailbreaking Black Box Large Language Models in Twenty Queries",
        
        description: "Red-Teaming 시리즈 #6 — LLM으로 LLM을 공격하는 자동 반복 정제 jailbreak 알고리즘, 20쿼리 (Chao et al., UPenn, 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/pair-attack/";
          
        },
      },{id: "post-universal-and-transferable-adversarial-attacks-on-aligned-language-models",
        
          title: "Universal and Transferable Adversarial Attacks on Aligned Language Models",
        
        description: "Red-Teaming 시리즈 #3 — Greedy Coordinate Gradient로 정렬된 LLM을 자동 공격하는 화이트박스 공격 (Zou et al., CMU, 2023)",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gcg-attack/";
          
        },
      },{id: "post-a-x-k1-technical-report",
        
          title: "A.X K1 Technical Report",
        
        description: "A.X K1 논문 리뷰 — 519B MoE 모델의 아키텍처, 데이터 파이프라인, Think-Fusion 학습 전략",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/ax-k1-technical-report/";
          
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
      },{id: "post-triton-07-flash-attention-3-triton으로-어디까지-가능한가",
        
          title: "Triton 07: Flash Attention 3 — Triton으로 어디까지 가능한가",
        
        description: "Hopper 전용인 Flash Attention 3를 Triton으로 어디까지 따라잡을 수 있는가 — 확장 autotune·persistent kernel·실패한 실험까지",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-07-flash-attention-v3/";
          
        },
      },{id: "post-triton-06-flash-attention-2-fa1-대비-5가지-최적화",
        
          title: "Triton 06: Flash Attention 2 — FA1 대비 5가지 최적화",
        
        description: "Flash Attention 2를 Triton으로 구현한다 — un-scaled 누적, exp2 트릭, Causal 2-stage, tl.dot accumulator, autotune",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/triton-06-flash-attention-v2/";
          
        },
      },{id: "post-triton-05-flash-attention-종합-프로젝트",
        
          title: "Triton 05: Flash Attention — 종합 프로젝트",
        
        description: "Flash Attention을 Triton으로 구현한다 — Forward/Backward 전체 구현과 RTX 4080·A100·H100·B200 아키텍처별 최적화 포인트",
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
          section: "News",},{id: "news-smaller-safer-stronger-sk-telecom-adaptiveml-s-gemma-3-4b-multilingual-moderation-model-has-been-featured-in-google-deepmind-s-gemmaverse-showcase-project-lead",
          title: 'Smaller, Safer, Stronger — SK Telecom × AdaptiveML’s Gemma 3 4B multilingual moderation...',
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
          section: "News",},{id: "teachings-data-science-fundamentals",
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
