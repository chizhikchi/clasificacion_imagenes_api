# 🎭 EfficientNetV2 Emotion Classification API

> **API REST para clasificación de emociones en tiempo real usando EfficientNetV2 y FastAPI**

Una solución containerizada de machine learning que clasifica emociones faciales en imágenes usando un modelo EfficientNetV2 optimizado, servido a través de una API FastAPI de alto rendimiento.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura Técnica](#️-arquitectura-técnica)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Evaluación del Modelo](#-evaluación-del-modelo)
- [Desarrollo Local](#-desarrollo-local)
- [Configuración de Producción](#-configuración-de-producción)
- [Rendimiento](#-rendimiento)
- [Troubleshooting](#-troubleshooting)
- [Limitaciones Conocidas](#-limitaciones-conocidas)

## ✨ Características

- **🚀 Alto Rendimiento**: API asíncrona con uvicorn ASGI server
- **🎯 Precisión**: Modelo EfficientNetV2 entrenado para 7 emociones
- **🐳 Containerizado**: Deployment listo con Docker
- **📊 Evaluación Integrada**: Endpoint para métricas de rendimiento
- **🔧 Robusto**: Manejo de errores y validación de entrada
- **📱 Fácil Integración**: API REST estándar con documentación automática

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   FastAPI App                           ││
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐││
│  │  │   uvicorn       │  │      EfficientNetV2            │││
│  │  │   ASGI Server   │  │      Model (GPU/CPU)           │││
│  │  └─────────────────┘  └─────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
    HTTP Requests                   Image Processing
    (Port 8080)                    (PIL + torchvision)
```

### Stack Tecnológico
- **Framework**: FastAPI + uvicorn
- **ML**: PyTorch + EfficientNetV2
- **Procesamiento**: PIL + torchvision transforms
- **Container**: Docker (python:3.9-slim)
- **Clasificación**: 7 emociones faciales

## 🚀 Quick Start

### Prerequisitos
- Docker instalado
- Archivo del modelo: `efficientnet_v2_model01.pth`

### 1. Construir el contenedor
```bash
docker build -t efficientnet-emotion-api .
```

### 2. Ejecutar la API
```bash
docker run -p 8080:8080 efficientnet-emotion-api
```

### 3. Verificar funcionamiento
```bash
curl http://localhost:8080/
```

### 4. Realizar predicción
```bash
curl -X POST -F "file=@imagen_test.jpg" http://localhost:8080/predict
```

## 📡 API Endpoints

### 🏠 Health Check
```http
GET /
```
**Respuesta:**
```json
{
  "message": "EfficientNetV2 Image Classification API"
}
```

### 🎯 Clasificación de Imagen
```http
POST /predict
Content-Type: multipart/form-data
```

**Parámetros:**
- `file`: Archivo de imagen (JPG, PNG, etc.)

**Respuesta Exitosa:**
```json
{
  "predicted_class": 3,
  "predicted_emotion": "happy",
  "confidence": 0.8947
}
```

**Emociones Soportadas:**
- `0`: angry (enojado)
- `1`: disgust (asco)
- `2`: fear (miedo)
- `3`: happy (feliz)
- `4`: neutral (neutral)
- `5`: sad (triste)
- `6`: surprise (sorpresa)

### 📊 Evaluación del Modelo
```http
POST /evaluate
Content-Type: application/json
```

**Parámetros:**
```json
{
  "validation_dir": "/app/validation"
}
```

**Respuesta:**
```json
{
  "performance": {
    "accuracy": 0.87,
    "macro_f1": 0.85,
    "micro_f1": 0.87,
    "confusion_matrix": [[...], [...]]
  },
  "efficiency": {
    "avg_inference_time_sec": 0.045,
    "throughput_img_per_sec": 22.2,
    "memory_usage_mb": 245.8,
    "model_size_mb": 85.4
  }
}
```

## 📈 Evaluación del Modelo

### Rendimiento Actual

El modelo de clasificación de emociones alcanza una precisión general del 70.8% sobre un conjunto de 7,066 imágenes distribuidas en 7 clases emocionales. Destaca especialmente en la detección de emociones de "felicidad" (F1: 0.89) y "sorpresa" (F1: 0.80), mientras que presenta mayor dificultad con "miedo" (F1: 0.53) y "tristeza" (F1: 0.59). El modelo procesa imágenes a una velocidad de 5.38 imágenes por segundo con un tiempo de inferencia promedio de 0.19 segundos, manteniendo un uso eficiente de memoria de 544 MB y un tamaño de modelo compacto de 78 MB, lo que lo hace adecuado para aplicaciones en tiempo real.

### Métricas Detalladas por Emoción

| Emoción | Precisión | Recall | F1-Score | Soporte |
|---------|-----------|--------|----------|---------|
| Angry | 59.6% | 67.7% | **63.4%** | 960 |
| Disgust | 72.5% | 71.2% | **71.8%** | 111 |
| Fear | 61.8% | 46.9% | **53.3%** | 1,018 |
| Happy | 89.5% | 88.9% | **89.2%** | 1,825 |
| Neutral | 63.7% | 71.5% | **67.3%** | 1,216 |
| Sad | 59.7% | 57.8% | **58.7%** | 1,139 |
| Surprise | 79.5% | 81.1% | **80.2%** | 797 |

### Evaluación Personalizada

Para evaluar el modelo con tu dataset de validación:

1. **Estructura de directorios:**
```
/validation/
├── angry/
│   ├── img1.jpg
│   └── img2.jpg
├── happy/
│   ├── img3.jpg
│   └── img4.jpg
└── ...
```

2. **Ejecutar evaluación:**
```bash
curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"validation_dir": "/path/to/validation"}'
```

## 💻 Uso sin Docker o Desarrollo

### Configurar entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar sin Docker
```bash
python app.py
```

### Acceder a documentación interactiva
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 🌐 Configuración de Producción

### Variables de entorno recomendadas
```bash
# Configuración del servidor
export HOST="0.0.0.0"
export PORT="8080"
export WORKERS="1"  # Para modelos ML, 1 worker es óptimo

# Configuración del modelo
export MODEL_PATH="efficientnet_v2_model01.pth"
export DEVICE="cuda"  # o "cpu"
```

### Escalado horizontal
```bash
# Múltiples instancias con load balancer
docker run -d -p 8081:8080 --name emotion-api-1 efficientnet-emotion-api
docker run -d -p 8082:8080 --name emotion-api-2 efficientnet-emotion-api
docker run -d -p 8083:8080 --name emotion-api-3 efficientnet-emotion-api
```

## ⚡ Rendimiento

### Métricas de Eficiencia
- **Throughput**: 5.38 imágenes por segundo
- **Latencia**: 186ms promedio por imagen
- **Memoria**: 544 MB en tiempo de ejecución
- **Tamaño del modelo**: 78 MB
- **Precisión general**: 70.8%

### Benchmarks de Hardware
- **CPU**: Optimizado para Intel/AMD x64
- **GPU**: Compatible con CUDA (mejora throughput ~3x)
- **RAM**: Mínimo 1GB recomendado para producción

## 🔧 Troubleshooting

### Problemas Comunes

**Error: "No module named 'torch'"**
```bash
# Verificar instalación de dependencias
pip install -r requirements.txt
```

**Error: "CUDA out of memory"**
```bash
# Forzar uso de CPU
export CUDA_VISIBLE_DEVICES=""
# O modificar en app.py: device = torch.device("cpu")
```

**Error: "Model file not found"**
```bash
# Verificar que el archivo del modelo existe
ls -la efficientnet_v2_model01.pth
# Verificar permisos
chmod 644 efficientnet_v2_model01.pth
```

**Container no responde**
```bash
# Verificar logs
docker logs <container_id>

# Verificar puerto
netstat -tulpn | grep 8080

# Testear desde dentro del container
docker exec -it <container_id> curl localhost:8080
```

### Logs y Debugging
```bash
# Ver logs en tiempo real
docker logs -f <container_id>

# Acceder al container para debugging
docker exec -it <container_id> /bin/bash
```

## ⚠️ Limitaciones Conocidas

### Técnicas
- **Tamaño de imagen**: Imágenes se redimensionan a 256x256px
- **Formato**: Solo acepta formatos estándar (JPG, PNG, BMP)
- **Concurrencia**: Optimizado para un worker (modelo stateful)

### Funcionales  
- **Detección facial**: No incluye detección previa de rostros
- **Múltiples caras**: Procesa toda la imagen, no caras individuales
- **Tiempo real**: No optimizado para video streaming
- **Batch processing**: Procesa una imagen por request
- **Emociones complejas**: Mayor dificultad con "miedo" y "tristeza"

## Estructura del Proyecto
```
.
├── app.py                          # Aplicación FastAPI principal
├── Dockerfile                      # Configuración del container
├── requirements.txt                # Dependencias Python
├── efficientnet_v2_model01.pth    # Modelo entrenado
├── README.md                       # Este archivo
└── validation/                     # Dataset de validación (opcional)
    ├── angry/
    ├── happy/
    └── ...
```

### Información del Sistema
- **Python**: 3.9+
- **PyTorch**: Compatible con CPU y CUDA
- **FastAPI**: Framework asíncrono
- **Docker**: Imagen base python:3.9-slim

---