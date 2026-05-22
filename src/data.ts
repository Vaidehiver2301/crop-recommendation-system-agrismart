export interface CropDataPoint {
  n: number;
  p: number;
  k: number;
  temperature: number;
  ph: number;
  rainfall: number;
  label: string;
}

export interface MetricSummary {
  min: number;
  max: number;
  avg: number;
}

// 10 classes of crops popular in Indian agronomy
export const AVAILABLE_CROPS = [
  "Rice",
  "Maize",
  "Chickpea",
  "Cotton",
  "Coffee",
  "Apple",
  "Grapes",
  "Mango",
  "Orange",
  "Coconut"
];

// Profile centers to generate a realistic deterministic dataset
const CROP_PROFILES: Record<string, {
  n: [number, number];
  p: [number, number];
  k: [number, number];
  temp: [number, number];
  ph: [number, number];
  rainfall: [number, number];
}> = {
  Rice: {
    n: [80, 100], p: [45, 55], k: [35, 45],
    temp: [22, 27], ph: [5.5, 6.5], rainfall: [170, 220]
  },
  Maize: {
    n: [70, 90], p: [40, 50], k: [15, 25],
    temp: [20, 28], ph: [5.6, 7.0], rainfall: [70, 110]
  },
  Chickpea: {
    n: [25, 40], p: [55, 65], k: [75, 85],
    temp: [16, 22], ph: [7.0, 8.0], rainfall: [40, 60]
  },
  Cotton: {
    n: [100, 120], p: [40, 50], k: [15, 25],
    temp: [26, 32], ph: [6.2, 7.5], rainfall: [60, 95]
  },
  Coffee: {
    n: [90, 110], p: [20, 30], k: [25, 35],
    temp: [22, 26], ph: [6.0, 6.8], rainfall: [140, 190]
  },
  Apple: {
    n: [15, 30], p: [120, 140], k: [195, 205],
    temp: [15, 21], ph: [5.5, 6.5], rainfall: [100, 125]
  },
  Grapes: {
    n: [20, 35], p: [120, 135], k: [190, 205],
    temp: [18, 24], ph: [5.5, 6.2], rainfall: [65, 80]
  },
  Mango: {
    n: [20, 35], p: [20, 30], k: [25, 35],
    temp: [27, 33], ph: [5.5, 6.8], rainfall: [85, 110]
  },
  Orange: {
    n: [15, 28], p: [8, 15], k: [15, 25],
    temp: [18, 30], ph: [6.0, 7.8], rainfall: [95, 125]
  },
  Coconut: {
    n: [15, 30], p: [8, 15], k: [25, 35],
    temp: [25, 30], ph: [5.5, 6.5], rainfall: [160, 230]
  }
};

// Seeded deterministic pseudo-random helper to make dataset fully reproducible
function seededRandom(seed: number) {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

export function generateDataset(): CropDataPoint[] {
  const dataset: CropDataPoint[] = [];
  let seed = 42;

  // Generate 12 samples per crop class = 120 points total
  AVAILABLE_CROPS.forEach((cropName) => {
    const profile = CROP_PROFILES[cropName];
    for (let i = 0; i < 12; i++) {
      const getVal = (range: [number, number]) => {
        const rnd = seededRandom(seed++);
        return Number((range[0] + rnd * (range[1] - range[0])).toFixed(1));
      };

      dataset.push({
        n: getVal(profile.n),
        p: getVal(profile.p),
        k: getVal(profile.k),
        temperature: getVal(profile.temp),
        ph: getVal(profile.ph),
        rainfall: getVal(profile.rainfall),
        label: cropName
      });
    }
  });

  return dataset;
}

export const DATASET_LIMIT_DEFAULTS = {
  n: { min: 0, max: 140, step: 1, label: "Nitrogen (N)", unit: "mg/kg", desc: "Primary nutrient for foliage growth" },
  p: { min: 5, max: 150, step: 1, label: "Phosphorus (P)", unit: "mg/kg", desc: "Supports root development & flowering" },
  k: { min: 5, max: 220, step: 1, label: "Potassium (K)", unit: "mg/kg", desc: "Improves disease resistance & crop quality" },
  temperature: { min: 10, max: 45, step: 0.1, label: "Temperature", unit: "°C", desc: "Ambient environmental temperature" },
  ph: { min: 3.5, max: 9.0, step: 0.1, label: "Soil pH", unit: "pH", desc: "Level of soil acidity or alkalinity" },
  rainfall: { min: 20, max: 300, step: 1, label: "Rainfall", unit: "mm", desc: "Water availability indicator" }
};

export const CROP_AGRONOMIC_METRICS: Record<string, string> = {
  Rice: "Rice requires high water retention and heavy clayey soils, thriving in waterlogged conditions with elevated Nitrogen levels.",
  Maize: "Maize performs best under deep, well-draining rich soils with medium rainfall and moderate nutrient enrichment.",
  Chickpea: "Chickpea (Gram) is a high-protein winter crop requiring airy, well-aerated soils, dry conditions, and extremely low Nitrogen.",
  Cotton: "Cotton grows optimally under high heat conditions, black cotton soil (regur) with moderate rainfall, and ample Nitrogen.",
  Coffee: "Coffee is traditionally cultivated along elevated slopes (hills) with deep, acidic, humid soils and high periodic rainfall.",
  Apple: "Apple cultivation is restricted to temperate and elevated cold Himalayan valleys, requiring a low-temperature winter soil profiling.",
  Grapes: "Grapes thrive on dry, warm conditions with rapid soil water drainage, needing generous potassium fertilizer.",
  Mango: "Mango is a tropical evergreen requiring warm temperatures, slightly acidic to neutral soils with moderate monsoon coverage.",
  Orange: "Oranges grow best under well-aerated loamy soils with optimal sun exposure and sensitive irrigation controls.",
  Coconut: "Coconut demands deep deltaic or tropical sandy coastlines, high ambient relative humidity, and consistent wet moisture."
};

export const PYTHON_STREAMLIT_CODE = `import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION & IN-MEMORY SYNTHETIC DATASET
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Crop Recommendation Engine",
    page_icon="🌱",
    layout="wide"
)

# Premium India-focused styled banner
st.markdown("""
<div style="background-color:#1e3d2f;padding:24px;border-radius:12px;margin-bottom:28px">
    <h1 style="color:#ffffff;margin:0;font-size:32px;font-family:'Segoe UI', sans-serif">🌱 Indian Crop Recommendation Engine</h1>
    <p style="color:#a8c3b4;margin:8px 0 0 0;font-size:16px">
        Determine the most optimal crop using <b>KNN</b>, <b>Decision Tree</b>, and <b>Random Forest</b> models.
    </p>
</div>
""", unsafe_allow_html=True)

# Build descriptive dataset centered on actual Indian soil conditions
@st.cache_data
def get_agronomic_dataset():
    crop_profiles = {
        "Rice": {"n": (80, 100), "p": (45, 55), "k": (35, 45), "temp": (22, 27), "ph": (5.5, 6.5), "rain": (170, 220)},
        "Maize": {"n": (70, 90), "p": (40, 50), "k": (15, 25), "temp": (20, 28), "ph": (5.6, 7.0), "rain": (70, 110)},
        "Chickpea": {"n": (25, 40), "p": (55, 65), "k": (75, 85), "temp": (16, 22), "ph": (7.0, 8.0), "rain": (40, 60)},
        "Cotton": {"n": (100, 120), "p": (40, 50), "k": (15, 25), "temp": (26, 32), "ph": (6.2, 7.5), "rain": (60, 95)},
        "Coffee": {"n": (90, 110), "p": (20, 30), "k": (25, 35), "temp": (22, 26), "ph": (6.0, 6.8), "rain": (140, 190)},
        "Apple": {"n": (15, 30), "p": (120, 140), "k": (195, 205), "temp": (15, 21), "ph": (5.5, 6.5), "rain": (100, 125)},
        "Grapes": {"n": (20, 35), "p": (120, 135), "k": (190, 205), "temp": (18, 24), "ph": (5.5, 6.2), "rain": (65, 80)},
        "Mango": {"n": (20, 35), "p": (20, 30), "k": (25, 35), "temp": (27, 33), "ph": (5.5, 6.8), "rain": (85, 110)},
        "Orange": {"n": (15, 28), "p": (8, 15), "k": (15, 25), "temp": (18, 30), "ph": (6.0, 7.8), "rain": (95, 125)},
        "Coconut": {"n": (15, 30), "p": (8, 15), "k": (25, 35), "temp": (25, 30), "ph": (5.5, 6.5), "rain": (160, 230)}
    }
    
    data = []
    np.random.seed(42)
    for crop, prof in crop_profiles.items():
        for _ in range(15):  # 15 samples per crop (150 samples total)
            row = {
                "N": np.random.uniform(prof["n"][0], prof["n"][1]),
                "P": np.random.uniform(prof["p"][0], prof["p"][1]),
                "K": np.random.uniform(prof["k"][0], prof["k"][1]),
                "Temperature": np.random.uniform(prof["temp"][0], prof["temp"][1]),
                "pH": np.random.uniform(prof["ph"][0], prof["ph"][1]),
                "Rainfall": np.random.uniform(prof["rain"][0], prof["rain"][1]),
                "Label": crop
            }
            data.append(row)
    return pd.DataFrame(data)

df = get_agronomic_dataset()

# ------------------------------------------------------------------
# 2. SEPARATE CHANNELS & TRAIN CLASSIFIERS
# ------------------------------------------------------------------
X = df[["N", "P", "K", "Temperature", "pH", "Rainfall"]]
y = df["Label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit normalizer (KNN is sensitive to feature scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: KNN
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train_scaled, y_train)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))

# Model 2: Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

# Model 3: Random Forest
rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# ------------------------------------------------------------------
# 3. INTERACTIVE LAYOUT DESIGN (SIDEBAR + MAIN)
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("🧪 Soil and Environmental Inputs")
    st.markdown("Adjust soil parameters to matching Indian agricultural conditions:")
    
    # User Inputs
    n = st.slider("Nitrogen (N) [mg/kg]", 0, 140, 85, help="Primary nutrient for foliage development.")
    p = st.slider("Phosphorus (P) [mg/kg]", 5, 150, 48, help="Supports root development and vigorous growth.")
    k = st.slider("Potassium (K) [mg/kg]", 5, 220, 42, help="Contributes to immune strength.")
    temp = st.slider("Temperature (°C)", 10.0, 45.0, 24.5, step=0.1, help="Ambient seasonal temperature.")
    ph = st.slider("Soil pH", 3.5, 9.0, 6.2, step=0.1, help="Neutral is ~7.0. Acidic is < 7. Alklaine is > 7.")
    rain = st.slider("Rainfall (mm)", 20, 300, 195, help="Monsoon indicators.")

    st.markdown("---")
    st.subheader("⚙️ Algorithm Hyperparameters")
    knn_k = st.slider("KNN Neighbors (K)", 1, 11, 5, step=2)
    dt_depth = st.slider("Decision Tree Max Depth", 2, 8, 5)

with col2:
    st.subheader("🌾 Optimal Crop Predictions")
    
    # Structure test point
    user_test = pd.DataFrame([[n, p, k, temp, ph, rain]], columns=["N", "P", "K", "Temperature", "pH", "Rainfall"])
    user_test_scaled = scaler.transform(user_test)
    
    # Update KNN if parameters changed
    if knn_k != 5:
        knn_model = KNeighborsClassifier(n_neighbors=knn_k)
        knn_model.fit(X_train_scaled, y_train)
        knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))
        
    # Update Decision Tree if depth changed
    if dt_depth != 5:
        dt_model = DecisionTreeClassifier(max_depth=dt_depth, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
        
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=dt_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    # Perform active predictions
    pred_knn = knn_model.predict(user_test_scaled)[0]
    pred_dt = dt_model.predict(user_test)[0]
    pred_rf = rf_model.predict(user_test)[0]

    # Metrics Box Layout
    pred1, pred2, pred3 = st.columns(3)
    
    with pred1:
        st.markdown(f"""
        <div style="background-color:#f1fdf4;border:1px solid #c2f0d1;padding:16px;border-radius:8px;text-align:center">
            <span style="font-size:12px;color:#1e3d2f"><b>K-NEAREST NEIGHBORS</b></span>
            <h3 style="color:#1e3d2f;margin:8px 0;font-size:20px">{pred_knn}</h3>
            <span style="font-size:11px;color:#4B5563">Val Accuracy: {knn_acc*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
    with pred2:
        st.markdown(f"""
        <div style="background-color:#fef8eb;border:1px solid #fce2b1;padding:16px;border-radius:8px;text-align:center">
            <span style="font-size:12px;color:#6b3e11"><b>DECISION TREE</b></span>
            <h3 style="color:#6b3e11;margin:8px 0;font-size:20px">{pred_dt}</h3>
            <span style="font-size:11px;color:#4B5563">Val Accuracy: {dt_acc*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
    with pred3:
        st.markdown(f"""
        <div style="background-color:#f3f4f6;border:1px solid #e5e7eb;padding:16px;border-radius:8px;text-align:center">
            <span style="font-size:12px;color:#374151"><b>RANDOM FOREST</b></span>
            <h3 style="color:#374151;margin:8px 0;font-size:20px">{pred_rf}</h3>
            <span style="font-size:11px;color:#4B5563">Val Accuracy: {rf_acc*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # 4. EXPLANATORY INSIGHTS & DESCRIPTION
    st.write("")
    st.subheader("💡 Agronomic Insights")
    
    unique_predictions = list(set([pred_knn, pred_dt, pred_rf]))
    
    for crop in unique_predictions:
        with st.expander(f"📖 About crop: {crop}"):
            crop_descriptions = {
                "Rice": "Rice requires heavy clayey soils, continuous waterlogging (high rainfall), and rich nitrogen input.",
                "Maize": "Maize performs best in warm climates with well-drained loamy soils and moderate, standard rainfall.",
                "Chickpea": "Chickpea is highly sensitive to nitrogen fertilizers (prefers low N) and requires low moisture, cool winter climates.",
                "Cotton": "Cotton demands long frost-free periods, high nitrogen fertilization, dry harvest climates, and clay-loam water retention.",
                "Coffee": "Coffee requires shaded altitudes, acidic soil rich in organics, and high temperate monsoons.",
                "Apple": "Apple is specific to cold temperate elevations with rich root depth drainage, potassium-rich and phosphorus-active nutrient pools.",
                "Grapes": "Grapes perform beautifully on fast-drained soils, long warm days, high potassium, and minimal standing monsoon water.",
                "Mango": "Mango requires tropical monsoon ranges, warm summers to sweeten fruits, and neutral soils.",
                "Orange": "Orange is highly active to sandy-loam drainage and neutral soils, moderate monsoons, and moderate warmth.",
                "Coconut": "Coconut requires humid sea air, highly consistent sandy-coast warm soil beds, and heavy annual monsoon rainfall."
            }
            st.info(crop_descriptions.get(crop, "Optimal crop for the selected profile."))

    # Display live stats
    st.markdown("---")
    st.subheader("📊 Dataset Distribution (First 10 records shown)")
    st.dataframe(df.head(10), use_container_width=True)
`;
