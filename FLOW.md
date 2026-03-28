Input Image
   ↓
Face Detection
   ↓
Face Alignment
   ↓
Feature Embedding (Deep Network)
   ↓
Similarity Matching
   ↓
Verification / Identification

1️⃣ Face Recognition System (Very Likely Question)

Interviewers often ask:

“Design a face recognition system used for identity verification.”

Typical Pipeline
6
Step-by-Step Pipeline

1. Face Detection

Detect face in image.

Common models

MTCNN
RetinaFace
YOLO-face

Output:

Bounding box
Facial landmarks

2. Face Alignment

Normalize face orientation using landmarks.

Example landmarks:

left eye
right eye
nose
mouth corners

This reduces variance from pose and rotation.

3. Face Embedding Extraction

Use deep model to generate face embedding vector.

Typical dimension

128
256
512

Example models

FaceNet
ArcFace
CosFace

Output example

embedding = [0.23, -0.45, 0.89, ....] (512D vector)

4. Matching

Compare embeddings using similarity.

Common metrics:

Cosine similarity

sim = (A · B) / (||A|| ||B||)

or

Euclidean distance

Decision:

if similarity > threshold
    same person
else
    different person

5. Liveness Detection (VERY important for Jumio)

Prevent spoofing.

Examples:

printed photo
phone replay attack
deepfake

Methods:

RGB + IR cameras
Blink detection
Depth estimation
Texture analysis
2️⃣ Bias & Fairness (Very Important)

They explicitly mentioned fairness and disparate impact, so expect questions like:

“How do you measure bias in a face recognition system?”

Sources of Bias
Dataset imbalance

Example:

80% light skin
10% dark skin
10% others

Model learns biased features.

Lighting conditions

Dark skin faces often poorly detected.

Camera sensor differences

Different mobile devices.

Metrics to Detect Bias

Evaluate metrics per demographic group.

Example groups:

Gender
Skin tone
Age
Ethnicity

Metrics:

FAR (False Accept Rate)
FRR (False Reject Rate)
TPR / FPR

Example bias detection:

FRR (light skin) = 1%
FRR (dark skin) = 8%

→ model biased.

Mitigation Techniques

1. Dataset balancing

Oversample underrepresented groups.

2. Demographic-aware evaluation

Separate validation sets.

3. Threshold calibration

Different thresholds for groups.

4. Domain adaptation

Improve performance on minority groups.

3️⃣ Face Recognition Loss Functions

Very likely question.

Triplet Loss

Used in FaceNet.

Anchor
Positive
Negative

Goal:

distance(anchor, positive)
<
distance(anchor, negative)
ArcFace Loss (Industry Standard)

Adds angular margin in embedding space.

Benefits:

better separation
better generalization
4️⃣ ML Pipeline Design (Jumio will test this)

Typical question:

“How would you build an end-to-end ML pipeline for face verification?”

Architecture
Data Collection
      ↓
Data Labeling
      ↓
Data Versioning (DVC)
      ↓
Training Pipeline
      ↓
Model Evaluation
      ↓
Model Registry
      ↓
Deployment
      ↓
Monitoring

Tools:

Airflow
Kubeflow
MLflow
SageMaker
5️⃣ Multi-GPU Training

You may get:

“How do you scale training across multiple GPUs?”

Approaches

Data Parallelism

Each GPU processes a batch subset.

GPU1 -> batch 1
GPU2 -> batch 2
GPU3 -> batch 3

Gradients synchronized.

In PyTorch:

DistributedDataParallel (DDP)
Model Parallelism

Split model across GPUs.

Example:

GPU1 -> encoder
GPU2 -> decoder

Used for very large models.

6️⃣ AWS Deployment (Likely Question)

Typical deployment architecture:

Client
  ↓
API Gateway
  ↓
EKS / EC2
  ↓
Face Recognition Service
  ↓
Feature DB (Embeddings)

Embedding storage options:

Faiss
Milvus
Pinecone
7️⃣ Face Recognition Evaluation Metrics

Important biometrics metrics.

FAR — False Accept Rate

Unauthorized user accepted.

FRR — False Reject Rate

Authorized user rejected.

ROC Curve

Plot:

TPR vs FPR
EER (Equal Error Rate)

Point where:

FAR = FRR

Lower EER → better system.
