# Home_Assignment_2
# Student name - Susmitha Dodla(700765311)

## Question 1: Cloud Computing for Deep Learning

### (a) Elasticity & Scalability

#### Elasticity
- Dynamically adjusts resources to match workload requirements.
- Enhances cost-efficiency by scaling up during peak usage and scaling down when demand decreases.  
- **Example:** Allocating extra GPUs during deep learning model training and releasing them afterward.

#### Scalability
- Expands system capacity by increasing available resources.

**Types:**  
- **Vertical Scaling:** Enhancing a single instance’s power (e.g., adding CPU, GPU, or RAM).  
- **Horizontal Scaling:** Distributing the workload by adding more machines.  

**Example:** Deploying training across multiple GPUs or cloud nodes.

### (b) AWS SageMaker vs. Google Vertex AI vs. Azure ML Studio

#### **AWS SageMaker**
- Fully managed service with integrated Jupyter notebooks.
- Compatible with TensorFlow, PyTorch, MXNet, and Scikit-learn.
- Supports auto-scaling for model training and deployment.
- Works with Nvidia GPUs and AWS Inferentia for efficiency.
- Provides MLOps capabilities, including Pipelines, AutoML, and monitoring.  
- **Best for:** Large-scale enterprise AI applications.

#### **Google Vertex AI**
- All-in-one platform combining AutoML and custom ML models.
- Supports TensorFlow, PyTorch, and Scikit-learn.
- Leverages Kubernetes for scalable training.
- Offers GPU and TPU support for high-performance computing.
- Features include Vertex Pipelines and AI Explanations.  
- **Best for:** AI workflows with a strong focus on automation and data handling.

#### **Azure Machine Learning Studio**
- User-friendly, low-code/no-code ML development environment.
- Works with TensorFlow, PyTorch, ONNX, and Scikit-learn.
- Supports hyperparameter tuning and distributed training.
- Compatible with Nvidia GPUs and FPGAs for deep learning.
- Offers MLOps, CI/CD, and AutoML tools.  
- **Best for:** Microsoft ecosystem users (Azure, Power BI, Synapse).
---

# Question 2: Convolution Operations with Different Parameters 
## Overview
This script applies a 3×3 convolution kernel to a 5×5 input matrix using TensorFlow. The kernel emphasizes edge detection by highlighting differences between neighboring values.

## Observations
1. **Stride = 1, Padding = VALID**  
   - Output matrix shrinks as no padding is added.
   - Edge details are preserved but dimensions are reduced.

2. **Stride = 1, Padding = SAME**  
   - Output retains the original size due to padding.
   - Edge information is present but slightly blurred.

3. **Stride = 2, Padding = VALID**  
   - Output size significantly reduces due to larger stride.
   - Captures only certain regions of the input.

4. **Stride = 2, Padding = SAME**  
   - Output maintains a balanced size due to padding.
   - Preserves more spatial features compared to VALID padding.

## Answer (Beginner-Friendly)
- **Padding "VALID"** removes edges, reducing output size.  
- **Padding "SAME"** keeps the output size similar to input by adding padding.  
- **Larger strides (2 vs. 1)** skip elements, reducing resolution but increasing efficiency.  
- **Smaller strides (1)** capture finer details but keep more data.  

This experiment demonstrates how padding and strides impact convolution outputs in deep learning.


---

## Question 3: CNN Feature Extraction with Filters and Pooling

### **Task 1: Edge Detection Using Convolution**
#### **Steps:**
1. Load an image in grayscale.
2. Define Sobel filters to capture edges in horizontal and vertical directions.
3. Apply convolution using `cv2.filter2D()`.
4. Display both the original and processed images using Matplotlib.

### **Task 2: Max & Average Pooling**
#### **Steps:**
1. Generate a 4×4 matrix with values between 0 and 9.
2. Apply MaxPooling (chooses the highest value in each region) and AveragePooling (computes the mean value in each region).
3. Use a **2×2 pool size** with **stride 2**.
4. Print the original and processed matrices.

### **Task 3: Observations on Convolution Operations**
#### **Observation:**
- The given code applies a 3×3 kernel to a 5×5 input matrix using TensorFlow’s convolution function.
- The kernel acts as an edge detection filter, highlighting differences in neighboring values.

#### **Effect of Stride:**
- **Stride 1:** Produces a more detailed feature map.
- **Stride 2:** Reduces output size, retaining fewer details.

#### **Effect of Padding:**
- **VALID padding:** Crops the edges, reducing output dimensions.
- **SAME padding:** Maintains dimensions by adding zero-padding around the input.

This experiment demonstrates how different stride and padding values affect the output size and feature extraction in convolutional layers.

---

## Question 4: CNN Architectures - Implementation & Comparison

### **Task 1: Build AlexNet**
#### **Steps:**
1. **First Convolution Layer:**
   - 96 filters, 11×11 kernel, stride 4, ReLU activation.
   - Followed by 3×3 MaxPooling with stride 2.
2. **Second Convolution Layer:**
   - 256 filters, 5×5 kernel, ReLU activation.
   - 3×3 MaxPooling with stride 2.
3. **Third, Fourth, and Fifth Convolution Layers:**
   - 384, 384, and 256 filters, 3×3 kernels, ReLU activation.
   - 3×3 MaxPooling with stride 2 after the last layer.
4. **Flatten output into a 1D vector.**
5. **Fully Connected Layers:**
   - Two layers with 4096 neurons each, ReLU activation, and 50% Dropout.
6. **Output Layer:**
   - 10 neurons using softmax activation for classification.

**Executing the script will display the model summary (layer details, output shapes, and parameter counts).**

### **Task 2: Implement Residual Block & ResNet**
#### **Step 1: Create a Residual Block**
- Takes an input tensor and applies two Conv2D layers (64 filters, 3×3, ReLU).
- Includes a skip connection that adds the input tensor to the output.

#### **Step 2: Build a ResNet-like Model**
1. Initial Conv2D layer (64 filters, 7×7 kernel, stride 2).
2. Implement two residual blocks.
3. Flatten output and pass it through a Dense layer with 128 neurons.
4. Final layer uses softmax activation for classification.
5. Print the model summary

