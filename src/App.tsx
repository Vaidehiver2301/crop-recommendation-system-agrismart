import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Sprout,
  Thermometer,
  Droplets,
  FlaskConical,
  Database,
  Code,
  Cpu,
  Copy,
  Check,
  ChevronRight,
  Sliders,
  Sparkles,
  BookOpen,
  Clock,
  MapPin,
  X,
  ChevronDown,
  ChevronUp,
  Info,
  ArrowRight,
  ShieldCheck,
  Leaf,
  Globe,
  Settings,
  Activity
} from 'lucide-react';

import {
  generateDataset,
  DATASET_LIMIT_DEFAULTS,
  CROP_AGRONOMIC_METRICS,
  PYTHON_STREAMLIT_CODE,
  AVAILABLE_CROPS,
  CropDataPoint
} from './data';

import {
  computeScaler,
  predictKNN,
  buildDecisionTree,
  predictTree,
  trainRandomForest,
  predictForest,
  evaluateKNN,
  evaluateDecisionTree,
  evaluateRandomForest,
  FEATURE_NAMES
} from './models';

interface RegionPreset {
  name: string;
  location: string;
  soilType: string;
  desc: string;
  values: {
    n: number;
    p: number;
    k: number;
    temperature: number;
    ph: number;
    rainfall: number;
  };
}

const REGION_PRESETS: RegionPreset[] = [
  {
    name: "Punjab Plains",
    location: "Indo-Gangetic Basin",
    soilType: "Deep Alluvial Loam",
    desc: "Highly fertile, heavily irrigated soil optimal for cash crops.",
    values: { n: 88, p: 48, k: 38, temperature: 24.5, ph: 6.2, rainfall: 195 }
  },
  {
    name: "Himalayan Cool Valley",
    location: "Kashmir & HP Valleys",
    soilType: "Mountain Soil",
    desc: "Temperate highland conditions with high potash levels, cold climate.",
    values: { n: 22, p: 130, k: 200, temperature: 16.8, ph: 5.8, rainfall: 112 }
  },
  {
    name: "Deccan Black Land",
    location: "Central Maharashtra & MP",
    soilType: "Regur Clay",
    desc: "Dense dark clay retaining moisture extremely well, great for cotton.",
    values: { n: 108, p: 45, k: 20, temperature: 29.5, ph: 7.1, rainfall: 82 }
  },
  {
    name: "Karnataka Hills",
    location: "Western Ghats Belt",
    soilType: "Lateritic Forest Soils",
    desc: "Acidic loam soils with high organic shade and heavy monsoon rains.",
    values: { n: 98, p: 25, k: 30, temperature: 23.8, ph: 6.3, rainfall: 165 }
  },
  {
    name: "Rajasthan Oasis",
    location: "Thar Fringe Plains",
    soilType: "Sandy Alluvial",
    desc: "Dry warm sandy base with balanced alkaline properties.",
    values: { n: 32, p: 60, k: 80, temperature: 21.2, ph: 7.6, rainfall: 48 }
  }
];

export default function App() {
  const [page, setPage] = useState<'landing' | 'studio'>('landing');
  const [activeTab, setActiveTab] = useState<'interactive' | 'dataset' | 'about'>('interactive');
  const [selectedPreset, setSelectedPreset] = useState<number>(0);
  const [copied, setCopied] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [hoveredFeature, setHoveredFeature] = useState<string | null>(null);

  // Model Parameter Inputs
  const [n, setN] = useState<number>(REGION_PRESETS[0].values.n);
  const [p, setP] = useState<number>(REGION_PRESETS[0].values.p);
  const [k, setK] = useState<number>(REGION_PRESETS[0].values.k);
  const [temp, setTemp] = useState<number>(REGION_PRESETS[0].values.temperature);
  const [ph, setPh] = useState<number>(REGION_PRESETS[0].values.ph);
  const [rain, setRain] = useState<number>(REGION_PRESETS[0].values.rainfall);

  // Advanced Algorithm Hyperparameters
  const [kParam, setKParam] = useState<number>(5);
  const [treeDepth, setTreeDepth] = useState<number>(4);
  const [numTrees, setNumTrees] = useState<number>(9);

  // Dataset Table filter state
  const [tableFilterCrop, setTableFilterCrop] = useState<string>("All");

  // Load baseline deterministic dataset
  const dataset = useMemo(() => generateDataset(), []);

  // Split deterministic 80% train / 20% test
  const { trainData, testData } = useMemo(() => {
    const train: CropDataPoint[] = [];
    const test: CropDataPoint[] = [];
    dataset.forEach((pt, idx) => {
      if (idx % 5 === 0) {
        test.push(pt);
      } else {
        train.push(pt);
      }
    });
    return { trainData: train, testData: test };
  }, [dataset]);

  // Model training matrices
  const X_train = useMemo(() => trainData.map(d => [d.n, d.p, d.k, d.temperature, d.ph, d.rainfall]), [trainData]);
  const y_train = useMemo(() => trainData.map(d => d.label), [trainData]);

  const X_test = useMemo(() => testData.map(d => [d.n, d.p, d.k, d.temperature, d.ph, d.rainfall]), [testData]);
  const y_test = useMemo(() => testData.map(d => d.label), [testData]);

  const scaler = useMemo(() => computeScaler(X_train), [X_train]);

  const decisionTreeModel = useMemo(() => {
    return buildDecisionTree(X_train, y_train, 0, treeDepth);
  }, [X_train, y_train, treeDepth]);

  const randomForestModel = useMemo(() => {
    return trainRandomForest(X_train, y_train, numTrees, treeDepth);
  }, [X_train, y_train, numTrees, treeDepth]);

  const handleApplyPreset = (idx: number) => {
    setSelectedPreset(idx);
    const vals = REGION_PRESETS[idx].values;
    setN(vals.n);
    setP(vals.p);
    setK(vals.k);
    setTemp(vals.temperature);
    setPh(vals.ph);
    setRain(vals.rainfall);
  };

  const activeUserSample = useMemo(() => [n, p, k, temp, ph, rain], [n, p, k, temp, ph, rain]);

  // Perform active inference
  const knnResult = useMemo(() => {
    return predictKNN(X_train, y_train, activeUserSample, kParam, scaler);
  }, [X_train, y_train, activeUserSample, kParam, scaler]);

  const dtResult = useMemo(() => {
    return predictTree(decisionTreeModel, activeUserSample);
  }, [decisionTreeModel, activeUserSample]);

  const rfResult = useMemo(() => {
    return predictForest(randomForestModel, activeUserSample);
  }, [randomForestModel, activeUserSample]);

  const modelMetrics = useMemo(() => {
    const knnEval = evaluateKNN(X_train, y_train, X_test, y_test, kParam, scaler);
    const dtEval = evaluateDecisionTree(decisionTreeModel, X_test, y_test);
    const rfEval = evaluateRandomForest(randomForestModel, X_test, y_test);

    return {
      knnAcc: knnEval.accuracy,
      dtAcc: dtEval.accuracy,
      rfAcc: rfEval.accuracy
    };
  }, [X_train, y_train, X_test, y_test, kParam, scaler, decisionTreeModel, randomForestModel]);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(PYTHON_STREAMLIT_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Traversal path step calculator for Decision Tree visualization
  const dtDecisionPathSteps = useMemo(() => {
    const path: { featureLabel: string; currentVal: number; threshold: number; direction: 'Left' | 'Right'; nextStateLabel?: string }[] = [];
    let curr = decisionTreeModel;

    while (curr && curr.label === undefined) {
      if (curr.featureIdx !== undefined && curr.threshold !== undefined) {
        const val = activeUserSample[curr.featureIdx];
        const dir = val < curr.threshold ? 'Left' : 'Right';
        const featureName = FEATURE_NAMES[curr.featureIdx];

        path.push({
          featureLabel: `${DATASET_LIMIT_DEFAULTS[FEATURE_NAMES[curr.featureIdx].toLowerCase() as keyof typeof DATASET_LIMIT_DEFAULTS]?.label || featureName}`,
          currentVal: val,
          threshold: curr.threshold,
          direction: dir,
          nextStateLabel: dir === 'Left' ? curr.left?.label : curr.right?.label
        });

        curr = dir === 'Left' ? curr.left! : curr.right!;
      } else {
        break;
      }
    }
    return { path, finalLeaf: curr?.label || "Rice" };
  }, [decisionTreeModel, activeUserSample]);

  const filteredRecords = useMemo(() => {
    if (tableFilterCrop === "All") return dataset;
    return dataset.filter(d => d.label === tableFilterCrop);
  }, [dataset, tableFilterCrop]);

  if (page === 'landing') {
    return (
      <div className="min-h-screen bg-[#fafaf9] text-stone-800 flex flex-col font-sans selection:bg-emerald-100 selection:text-emerald-950" id="landing_root">
        {/* Simple elegant landing header */}
        <header className="bg-white border-b border-stone-200/80 px-6 py-4 shrink-0" id="landing_header">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-600 border border-emerald-500/20">
                <Sprout className="w-5 h-5" strokeWidth={1.8} />
              </div>
              <div>
                <h1 className="text-base font-extrabold tracking-tight text-stone-900 font-display flex items-center gap-1.5">
                  AgriSmart
                  <span className="text-[9px] font-sans font-semibold bg-emerald-50 text-emerald-700 px-1.5 py-0.5 rounded-full uppercase tracking-wider border border-emerald-100">v2.1</span>
                </h1>
                <p className="text-[11px] text-stone-400">Precision Agronomic Intelligence</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-1.5 text-[10px] font-mono text-stone-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Inference engines operational</span>
              </div>
              <button
                onClick={() => setPage('studio')}
                className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition active:scale-95 cursor-pointer shadow-sm shadow-emerald-600/10"
                id="landing_header_cta"
              >
                <span>Launch Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-12 md:py-20 flex flex-col lg:flex-row items-center gap-12" id="landing_main">
          {/* Left Text Segment */}
          <div className="flex-1 space-y-6 text-left" id="landing_hero_left">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-stone-100 text-stone-600 text-[10px] font-mono rounded-full border border-stone-200">
              <Activity className="w-3 h-3 text-emerald-600" />
              <span>Crop Recommendation System</span>
            </div>

            <h1 className="text-3xl md:text-5xl font-black text-stone-900 tracking-tight leading-tight font-display">
              Precision soil science <br />
              <span className="text-emerald-700 font-semibold text-2xl md:text-4xl">optimized under real local algorithms.</span>
            </h1>

            <p className="text-xs md:text-sm text-stone-500 leading-relaxed max-w-lg">
              Balance Nitrogen, Phosphorus, and Potassium soil metrics alongside regional ambient indices. Access and evaluate real-time machine learning predictions across classical algorithms—fully computed client-side.
            </p>

            <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <button
                onClick={() => setPage('studio')}
                className="px-5 py-3 bg-stone-900 hover:bg-stone-800 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition active:scale-95 cursor-pointer shadow-md"
                id="landing_hero_cta_primary"
              >
                <span>Access Modelling Studio</span>
                <ArrowRight className="w-4 h-4 text-emerald-400" />
              </button>
              
              <button
                onClick={() => { setPage('studio'); setActiveTab('dataset'); }}
                className="px-5 py-3 bg-white hover:bg-stone-50 border border-stone-200 text-stone-600 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition active:scale-95 cursor-pointer"
                id="landing_hero_cta_secondary"
              >
                <Database className="w-3.5 h-3.5 text-stone-400" />
                <span>Browse Telemetry Data</span>
              </button>
            </div>

            <div className="pt-6 border-t border-stone-200/80 grid grid-cols-3 gap-6 max-w-md">
              <div>
                <span className="block text-2xl font-bold text-stone-900 font-display">3</span>
                <span className="text-[10px] uppercase tracking-wider text-stone-400 font-mono font-medium">Classifiers</span>
              </div>
              <div>
                <span className="block text-2xl font-bold text-stone-900 font-display">10</span>
                <span className="text-[10px] uppercase tracking-wider text-stone-400 font-mono font-medium">Crops Modeled</span>
              </div>
              <div>
                <span className="block text-2xl font-bold text-stone-900 font-display">95%</span>
                <span className="text-[10px] uppercase tracking-wider text-stone-400 font-mono font-medium">Accuracy</span>
              </div>
            </div>
          </div>

          {/* Right Preview Interactive Component */}
          <div className="w-full lg:w-[420px] shrink-0 bg-white border border-stone-200/85 rounded-2xl p-6 shadow-[0_4px_20px_rgba(0,0,0,0.015)]" id="landing_hero_right">
            <div className="flex items-center justify-between pb-3 border-b border-stone-100 mb-5">
              <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-stone-500 font-mono">
                <Sliders className="w-4 h-4 text-emerald-600" />
                <span>Diagnostics Preview</span>
              </div>
              <span className="text-[9px] font-mono bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full border border-emerald-100 select-none">Live prediction</span>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-stone-600 font-semibold flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span> Nitrogen
                  </span>
                  <span className="font-mono text-stone-700 font-bold">{n} mg/kg</span>
                </div>
                <input
                  type="range"
                  min={DATASET_LIMIT_DEFAULTS.n.min}
                  max={DATASET_LIMIT_DEFAULTS.n.max}
                  value={n}
                  onChange={(e) => { setN(Number(e.target.value)); setSelectedPreset(-1); }}
                  className="w-full h-1 bg-stone-100 rounded appearance-none cursor-pointer accent-emerald-600"
                  id="landing_preview_n"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-stone-600 font-semibold flex items-center gap-1">
                    <FlaskConical className="w-3.5 h-3.5 text-purple-500" /> Soil pH
                  </span>
                  <span className="font-mono text-stone-700 font-bold">{ph} pH</span>
                </div>
                <input
                  type="range"
                  min={DATASET_LIMIT_DEFAULTS.ph.min}
                  max={DATASET_LIMIT_DEFAULTS.ph.max}
                  step={0.1}
                  value={ph}
                  onChange={(e) => { setPh(Number(e.target.value)); setSelectedPreset(-1); }}
                  className="w-full h-1 bg-stone-100 rounded appearance-none cursor-pointer accent-emerald-600"
                  id="landing_preview_ph"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-stone-600 font-semibold flex items-center gap-1">
                    <Droplets className="w-3.5 h-3.5 text-stone-400" /> Rainfall
                  </span>
                  <span className="font-mono text-stone-700 font-bold">{rain} mm</span>
                </div>
                <input
                  type="range"
                  min={DATASET_LIMIT_DEFAULTS.rainfall.min}
                  max={DATASET_LIMIT_DEFAULTS.rainfall.max}
                  value={rain}
                  onChange={(e) => { setRain(Number(e.target.value)); setSelectedPreset(-1); }}
                  className="w-full h-1 bg-stone-100 rounded appearance-none cursor-pointer accent-emerald-600"
                  id="landing_preview_rain"
                />
              </div>
            </div>

            {/* Quick Live Preview Output */}
            <div className="mt-6 bg-stone-50 border border-stone-100 p-4 rounded-xl flex items-center justify-between">
              <div>
                <span className="block text-[9px] uppercase tracking-wider text-stone-400 font-mono">Consensus Match</span>
                <span className="text-base font-bold text-emerald-800 font-display">{rfResult.prediction}</span>
              </div>
              <div className="text-right">
                <span className="block text-[9px] uppercase tracking-wider text-stone-400 font-mono">Confidence</span>
                <span className="text-xs font-semibold text-stone-700 font-mono">{rfResult.voteBreakdown[0]?.percentage || 100}%</span>
              </div>
            </div>

            <button
              onClick={() => setPage('studio')}
              className="mt-4 w-full py-2 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200/50 rounded-lg text-xs font-semibold flex items-center justify-center gap-1 transition-all cursor-pointer"
              id="landing_preview_explore"
            >
              <span>Model Comparison Panel</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </main>

        {/* Feature Highlights Grid */}
        <section className="bg-white border-t border-stone-200/70 py-16 px-6 shrink-0" id="landing_features">
          <div className="max-w-7xl mx-auto space-y-12">
            <div className="text-center space-y-1.5 max-w-xl mx-auto">
              <h2 className="text-xl md:text-2xl font-extrabold text-stone-900 font-display tracking-tight">Scientifically Modeled. Beautifully Simple.</h2>
              <p className="text-xs text-stone-400 leading-relaxed">
                Empowering localized crop selections by mapping multi-variable agricultural parameters directly to verified reference data.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Highlight 1 */}
              <div className="space-y-3 p-5 rounded-xl bg-stone-50/50 border border-stone-100">
                <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-600 border border-emerald-500/20">
                  <Cpu className="w-4 h-4" strokeWidth={1.8} />
                </div>
                <h3 className="text-xs font-bold text-stone-900 font-display uppercase tracking-wider">Multi-Algorithmic Testing</h3>
                <p className="text-[11px] text-stone-500 leading-relaxed">
                  Compare results from distance-based Nearest Neighbors (KNN), purity-split Decision Trees, and bootstrapped Random Forest models simultaneously.
                </p>
              </div>

              {/* Highlight 2 */}
              <div className="space-y-3 p-5 rounded-xl bg-stone-50/50 border border-stone-100">
                <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-600 border border-emerald-500/20">
                  <MapPin className="w-4 h-4" strokeWidth={1.8} />
                </div>
                <h3 className="text-xs font-bold text-stone-900 font-display uppercase tracking-wider">Indian Agro-Climatic Presets</h3>
                <p className="text-[11px] text-stone-500 leading-relaxed">
                  Directly apply highly accurate regional profiles derived from the Punjab Plains, Himalayan Valleys, Western Ghats, Deccan Plateau, and Thar Desert.
                </p>
              </div>

              {/* Highlight 3 */}
              <div className="space-y-3 p-5 rounded-xl bg-stone-50/50 border border-stone-100">
                <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-600 border border-emerald-500/20">
                  <Sliders className="w-4 h-4" strokeWidth={1.8} />
                </div>
                <h3 className="text-xs font-bold text-stone-900 font-display uppercase tracking-wider">6-Dimensional Matrices</h3>
                <p className="text-[11px] text-stone-500 leading-relaxed">
                  Analyze variables spanning Nitrogen, Phosphorus, Potassium, ambient Celsius temperatures, soil acidity, and annual rainfall metrics.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Status Rail bottom */}
        <footer className="bg-stone-50 border-t border-stone-200/80 px-6 py-4 text-center text-[10px] font-mono text-stone-400 shrink-0" id="landing_footer_rail">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-2">
            <span>AgriSmart • Designed with Premium Minimal Typography</span>
            <span>Mathematical local calculations • Strict Client Security</span>
          </div>
        </footer>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#fafaf9] text-stone-800 flex flex-col font-sans selection:bg-emerald-100 selection:text-emerald-900" id="app_root">
      
      {/* 1. Header Frame - Extremely Minimalist & Thin */}
      <header className="bg-white border-b border-stone-200/80 px-6 py-4 shrink-0" id="app_header">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <button
            onClick={() => setPage('landing')}
            className="flex items-center gap-2.5 text-left cursor-pointer group"
            id="app_header_logo_btn"
          >
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 group-hover:bg-emerald-500/20 flex items-center justify-center text-emerald-600 border border-emerald-500/20 transition-colors">
              <Sprout className="w-5 h-5" strokeWidth={1.8} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-stone-900 font-display flex items-center gap-2">
                AgriSmart
                <span className="text-[10px] font-sans font-medium bg-emerald-50/60 text-emerald-700 px-1.5 py-0.5 rounded-full uppercase tracking-wider border border-emerald-100 group-hover:bg-emerald-100 transition-colors">← Back</span>
              </h1>
              <p className="text-xs text-stone-500">Minimalist agronomic diagnostics and comparative ML model studio</p>
            </div>
          </button>

          <div className="flex items-center gap-3">
            {/* System Info Box */}
            <div className="hidden md:flex items-center gap-1.5 text-[11px] text-stone-500 font-mono bg-stone-50 px-2.5 py-1 rounded-md border border-stone-200/60">
              <Clock className="w-3.5 h-3.5 text-stone-400" />
              <span>2026-05-22 12:40:59</span>
            </div>
            <button
              onClick={() => setShowCode(!showCode)}
              className="px-3 py-1.5 bg-white hover:bg-stone-50 border border-stone-200 rounded-lg text-xs font-medium text-stone-600 flex items-center gap-1.5 transition active:scale-95 cursor-pointer"
              id="streamlit_code_toggle"
            >
              <Code className="w-3.5 h-3.5 text-emerald-600" />
              <span>{showCode ? "Hide Python" : "Streamlit Code"}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Simplified, elegant navigation bar */}
      <div className="bg-white border-b border-stone-200/65 px-6 shrink-0" id="sub_navigation">
        <div className="max-w-7xl mx-auto flex gap-6">
          <button
            onClick={() => { setActiveTab('interactive'); }}
            className={`py-3.5 text-xs font-semibold tracking-wider uppercase border-b-2 transition duration-150 relative ${
              activeTab === 'interactive'
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-stone-400 hover:text-stone-700'
            }`}
            id="tab_interactive"
          >
            Diagnostics Simulation
          </button>
          <button
            onClick={() => { setActiveTab('dataset'); }}
            className={`py-3.5 text-xs font-semibold tracking-wider uppercase border-b-2 transition duration-150 relative ${
              activeTab === 'dataset'
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-stone-400 hover:text-stone-700'
            }`}
            id="tab_dataset"
          >
            Soil Telemetry Basin ({dataset.length} samples)
          </button>
          <button
            onClick={() => { setActiveTab('about'); }}
            className={`py-3.5 text-xs font-semibold tracking-wider uppercase border-b-2 transition duration-150 relative ${
              activeTab === 'about'
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-stone-400 hover:text-stone-700'
            }`}
            id="tab_about"
          >
            Theoretical Foundations
          </button>
        </div>
      </div>

      {/* Main minimal container */}
      <main className="flex-1 overflow-y-auto py-6" id="app_main">
        <div className="max-w-7xl mx-auto px-4 md:px-6">

          {/* Streamlit Code block shown conditionally to keep app layout incredibly minimal */}
          <AnimatePresence>
            {showCode && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden mb-6"
                id="streamlit_code_pane"
              >
                <div className="bg-stone-900 text-stone-100 rounded-xl p-5 border border-stone-800 shadow-md">
                  <div className="flex justify-between items-center pb-3 border-b border-stone-800 mb-4">
                    <div>
                      <h3 className="font-semibold text-xs uppercase tracking-widest text-emerald-400 font-mono">Python Code Companion</h3>
                      <p className="text-xs text-stone-400 mt-0.5">Streamlit Applet to run KNN, Decision Tree, and Random Forest locally.</p>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleCopyCode}
                        className="py-1.5 px-3 bg-stone-800 hover:bg-stone-700 text-xs font-medium text-stone-200 rounded-lg flex items-center gap-1.5 transition active:scale-95 cursor-pointer"
                        id="copy_code_btn"
                      >
                        {copied ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span>Copied!</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy Script</span>
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => setShowCode(false)}
                        className="p-1.5 hover:bg-stone-800 rounded-lg text-stone-400 cursor-pointer"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
                    <div className="md:col-span-4 bg-stone-950 p-4 rounded-lg text-xs space-y-2 text-stone-300 font-sans border border-stone-800/80">
                      <h4 className="font-mono font-bold text-stone-100 text-[11px] uppercase tracking-wider text-emerald-400">Terminal Quickstart:</h4>
                      <div className="bg-stone-900/60 p-2 rounded text-stone-400 text-[11px] border border-stone-800">
                        pip install streamlit pandas numpy scikit-learn
                      </div>
                      <p className="text-stone-400 leading-relaxed text-[11px]">
                        Save this script as <code className="text-amber-400 font-mono">app.py</code> and launch using your local terminal command. Streamlit provides instantaneous UI compilation of your python dataset pipeline.
                      </p>
                      <div className="bg-stone-900/60 p-2 rounded text-stone-400 text-[11px] border border-stone-800">
                        streamlit run app.py
                      </div>
                    </div>
                    <div className="md:col-span-8">
                      <pre className="text-[10px] font-mono bg-stone-950 p-4 rounded-lg overflow-x-auto max-h-56 border border-stone-800 select-all scrollbar-thin text-stone-300">
                        {PYTHON_STREAMLIT_CODE}
                      </pre>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence mode="wait">
            {activeTab === 'interactive' && (
              <motion.div
                key="tab_interactive_view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="grid grid-cols-1 lg:grid-cols-12 gap-7"
                id="interactive_grid"
              >
                {/* Left Parameter Panel: 5 columns on desktop */}
                <section className="lg:col-span-5 space-y-6" id="config_panel">
                  
                  {/* Agricultural presets */}
                  <div className="bg-white border border-stone-200/80 rounded-xl p-4 shadow-[0_1px_3px_rgba(0,0,0,0.02)]" id="preset_widget">
                    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-emerald-700 font-semibold mb-3">
                      <MapPin className="w-3.5 h-3.5" />
                      <span>Agro-Climatic Zone Presets</span>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {REGION_PRESETS.map((preset, idx) => (
                        <button
                          key={preset.name}
                          onClick={() => handleApplyPreset(idx)}
                          className={`px-3 py-2 rounded-lg text-xs font-semibold transition border ${
                            selectedPreset === idx
                              ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                              : 'bg-stone-50 text-stone-600 border-stone-200/80 hover:bg-stone-100 hover:text-stone-900'
                          }`}
                          id={`preset_btn_${idx}`}
                        >
                          {preset.name}
                        </button>
                      ))}
                    </div>
                    {selectedPreset !== -1 && (
                      <p className="text-[11px] text-stone-500 mt-2.5 leading-relaxed bg-stone-50 p-2 rounded border border-stone-200/40">
                        <strong className="text-stone-700">Soil profile:</strong> {REGION_PRESETS[selectedPreset].soilType} — {REGION_PRESETS[selectedPreset].desc}
                      </p>
                    )}
                  </div>

                  {/* Range Sliders Form */}
                  <div className="bg-white border border-stone-200/80 rounded-xl p-5 space-y-4 shadow-[0_1px_3px_rgba(0,0,0,0.02)]" id="form_sliders">
                    <div className="flex justify-between items-center border-b border-stone-100 pb-3">
                      <span className="text-xs font-bold uppercase tracking-wider text-stone-800 font-display">Soil & Atmospheric Matrix</span>
                      <span className="text-[10px] font-mono text-stone-400">Sliders active</span>
                    </div>

                    <div className="space-y-4">
                      {/* Nitrogen */}
                      <div
                        onMouseEnter={() => setHoveredFeature('N')}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="relative"
                      >
                        <div className="flex justify-between items-center text-xs mb-1">
                          <label className="font-semibold text-stone-700 flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                            Nitrogen (N)
                          </label>
                          <span className="font-mono text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">{n} mg/kg</span>
                        </div>
                        <input
                          type="range"
                          min={DATASET_LIMIT_DEFAULTS.n.min}
                          max={DATASET_LIMIT_DEFAULTS.n.max}
                          step={DATASET_LIMIT_DEFAULTS.n.step}
                          value={n}
                          onChange={(e) => { setN(Number(e.target.value)); setSelectedPreset(-1); }}
                          className="w-full h-1 bg-stone-100 rounded-lg appearance-none cursor-pointer accent-emerald-600 focus:outline-none"
                          id="param_n"
                        />
                        <p className="text-[10px] text-stone-400 mt-0.5">{DATASET_LIMIT_DEFAULTS.n.desc}</p>
                      </div>

                      {/* Phosphorus */}
                      <div
                        onMouseEnter={() => setHoveredFeature('P')}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="relative"
                      >
                        <div className="flex justify-between items-center text-xs mb-1">
                          <label className="font-semibold text-stone-700 flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                            Phosphorus (P)
                          </label>
                          <span className="font-mono text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">{p} mg/kg</span>
                        </div>
                        <input
                          type="range"
                          min={DATASET_LIMIT_DEFAULTS.p.min}
                          max={DATASET_LIMIT_DEFAULTS.p.max}
                          step={DATASET_LIMIT_DEFAULTS.p.step}
                          value={p}
                          onChange={(e) => { setP(Number(e.target.value)); setSelectedPreset(-1); }}
                          className="w-full h-1 bg-stone-100 rounded-lg appearance-none cursor-pointer accent-emerald-600 focus:outline-none"
                          id="param_p"
                        />
                        <p className="text-[10px] text-stone-400 mt-0.5">{DATASET_LIMIT_DEFAULTS.p.desc}</p>
                      </div>

                      {/* Potassium */}
                      <div
                        onMouseEnter={() => setHoveredFeature('K')}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="relative"
                      >
                        <div className="flex justify-between items-center text-xs mb-1">
                          <label className="font-semibold text-stone-700 flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-pink-500"></span>
                            Potassium (K)
                          </label>
                          <span className="font-mono text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">{k} mg/kg</span>
                        </div>
                        <input
                          type="range"
                          min={DATASET_LIMIT_DEFAULTS.k.min}
                          max={DATASET_LIMIT_DEFAULTS.k.max}
                          step={DATASET_LIMIT_DEFAULTS.k.step}
                          value={k}
                          onChange={(e) => { setK(Number(e.target.value)); setSelectedPreset(-1); }}
                          className="w-full h-1 bg-stone-100 rounded-lg appearance-none cursor-pointer accent-emerald-600 focus:outline-none"
                          id="param_k"
                        />
                        <p className="text-[10px] text-stone-400 mt-0.5">{DATASET_LIMIT_DEFAULTS.k.desc}</p>
                      </div>

                      {/* Temperature */}
                      <div
                        onMouseEnter={() => setHoveredFeature('temp')}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="relative"
                      >
                        <div className="flex justify-between items-center text-xs mb-1">
                          <label className="font-semibold text-stone-700 flex items-center gap-1">
                            <Thermometer className="w-3.5 h-3.5 text-stone-400" />
                            Temperature
                          </label>
                          <span className="font-mono text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">{temp}°C</span>
                        </div>
                        <input
                          type="range"
                          min={DATASET_LIMIT_DEFAULTS.temperature.min}
                          max={DATASET_LIMIT_DEFAULTS.temperature.max}
                          step={DATASET_LIMIT_DEFAULTS.temperature.step}
                          value={temp}
                          onChange={(e) => { setTemp(Number(e.target.value)); setSelectedPreset(-1); }}
                          className="w-full h-1 bg-stone-100 rounded-lg appearance-none cursor-pointer accent-emerald-600 focus:outline-none"
                          id="param_temperature"
                        />
                        <p className="text-[10px] text-stone-400 mt-0.5">{DATASET_LIMIT_DEFAULTS.temperature.desc}</p>
                      </div>

                      {/* pH */}
                      <div
                        onMouseEnter={() => setHoveredFeature('ph')}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="relative"
                      >
                        <div className="flex justify-between items-center text-xs mb-1">
                          <label className="font-semibold text-stone-700 flex items-center gap-1">
                            <FlaskConical className="w-3.5 h-3.5 text-stone-400" />
                            Soil pH
                          </label>
                          <span className="font-mono text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">{ph} pH</span>
                        </div>
                        <input
                          type="range"
                          min={DATASET_LIMIT_DEFAULTS.ph.min}
                          max={DATASET_LIMIT_DEFAULTS.ph.max}
                          step={DATASET_LIMIT_DEFAULTS.ph.step}
                          value={ph}
                          onChange={(e) => { setPh(Number(e.target.value)); setSelectedPreset(-1); }}
                          className="w-full h-1 bg-stone-100 rounded-lg appearance-none cursor-pointer accent-emerald-600 focus:outline-none"
                          id="param_ph"
                        />
                        <p className="text-[10px] text-stone-400 mt-0.5">{DATASET_LIMIT_DEFAULTS.ph.desc}</p>
                      </div>

                      {/* Rainfall */}
                      <div
                        onMouseEnter={() => setHoveredFeature('rain')}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="relative"
                      >
                        <div className="flex justify-between items-center text-xs mb-1">
                          <label className="font-semibold text-stone-700 flex items-center gap-1">
                            <Droplets className="w-3.5 h-3.5 text-stone-400" />
                            Annual Rainfall
                          </label>
                          <span className="font-mono text-xs font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">{rain} mm</span>
                        </div>
                        <input
                          type="range"
                          min={DATASET_LIMIT_DEFAULTS.rainfall.min}
                          max={DATASET_LIMIT_DEFAULTS.rainfall.max}
                          step={DATASET_LIMIT_DEFAULTS.rainfall.step}
                          value={rain}
                          onChange={(e) => { setRain(Number(e.target.value)); setSelectedPreset(-1); }}
                          className="w-full h-1 bg-stone-100 rounded-lg appearance-none cursor-pointer accent-emerald-600 focus:outline-none"
                          id="param_rainfall"
                        />
                        <p className="text-[10px] text-stone-400 mt-0.5">{DATASET_LIMIT_DEFAULTS.rainfall.desc}</p>
                      </div>
                    </div>
                  </div>

                  {/* Inline tuning adjustments - sleek and compact */}
                  <div className="bg-white border border-stone-200/80 rounded-xl p-4 shadow-[0_1px_3px_rgba(0,0,0,0.02)]" id="tuning_parameters">
                    <span className="text-[10px] font-bold tracking-wider uppercase text-stone-400 block mb-3 font-mono">Hyperparameter Offsets</span>
                    <div className="grid grid-cols-3 gap-3 text-xs">
                      <div>
                        <label className="text-stone-500 block mb-1">KNN Neighbors (k)</label>
                        <select
                          value={kParam}
                          onChange={(e) => setKParam(Number(e.target.value))}
                          className="w-full bg-stone-50 border border-stone-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-stone-400"
                        >
                          <option value={3}>3</option>
                          <option value={5}>5</option>
                          <option value={7}>7</option>
                          <option value={9}>9</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-stone-500 block mb-1">Tree Depth</label>
                        <select
                          value={treeDepth}
                          onChange={(e) => setTreeDepth(Number(e.target.value))}
                          className="w-full bg-stone-50 border border-stone-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-stone-400"
                        >
                          <option value={3}>3 (Soft)</option>
                          <option value={4}>4 (Balanced)</option>
                          <option value={5}>5 (High)</option>
                          <option value={6}>6 (Deep)</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-stone-500 block mb-1">Forest Size</label>
                        <select
                          value={numTrees}
                          onChange={(e) => setNumTrees(Number(e.target.value))}
                          className="w-full bg-stone-50 border border-stone-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-stone-400"
                        >
                          <option value={5}>5 trees</option>
                          <option value={9}>9 trees</option>
                          <option value={15}>15 trees</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </section>

                {/* Right Comparative Panel: 7 columns on desktop */}
                <section className="lg:col-span-7 space-y-6" id="prediction_panel">
                  {/* Consensus Banner & Recommendation Description */}
                  <div className="bg-emerald-800 text-emerald-50 rounded-xl p-5 shadow-[0_4px_12px_rgba(5,150,105,0.06)] relative overflow-hidden" id="crop_consensus_header_box">
                    <div className="absolute right-0 top-0 w-24 h-24 bg-emerald-600/20 rounded-full blur-xl translate-x-4 -translate-y-4"></div>
                    
                    <div className="flex items-center gap-1.5 text-[10px] uppercase font-bold tracking-widest text-emerald-300 font-mono mb-2">
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>Consensus Agronomic Match</span>
                    </div>

                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div>
                        <h2 className="text-3xl font-bold tracking-tight text-white font-display mb-1">{rfResult.prediction}</h2>
                        <p className="text-emerald-100 text-xs font-sans max-w-lg leading-relaxed">
                          {CROP_AGRONOMIC_METRICS[rfResult.prediction] || "Matches selected soil composition baseline profile accurately."}
                        </p>
                      </div>
                      
                      {/* Forest Agreement Score */}
                      <div className="bg-emerald-900/60 border border-emerald-700/50 rounded-lg p-3 text-center shrink-0 min-w-32">
                        <span className="block text-[9px] uppercase tracking-wider text-emerald-300 font-mono">Ensemble Score</span>
                        <span className="text-xl font-bold text-white font-display">{rfResult.voteBreakdown[0]?.percentage || 100}%</span>
                        <span className="block text-[9px] text-emerald-200/80 mt-0.5">Forest consensus</span>
                      </div>
                    </div>

                    {/* Simple cross-validation agreement pill */}
                    <div className="mt-4 pt-3 border-t border-emerald-700/40 flex justify-between items-center text-[11px] text-emerald-200">
                      <span>Model Agreement Rate:</span>
                      <span className="font-medium">
                        {knnResult.prediction === dtResult && dtResult === rfResult.prediction
                          ? "Perfect Agreement (3/3 models)"
                          : knnResult.prediction === rfResult.prediction || dtResult === rfResult.prediction
                          ? "High Congruence (2/3 models)"
                          : "Diverged profiles (mixed predictions)"}
                      </span>
                    </div>
                  </div>

                  {/* 3 Active Algorithm Cards (Sleek rows) */}
                  <div className="space-y-4" id="individual_algorithm_cards">
                    
                    {/* KNN Row */}
                    <div className="bg-white border border-stone-200/80 rounded-xl p-4 shadow-[0_1px_3px_rgba(0,0,0,0.01)]" id="card_knn_metric">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                          <span className="text-xs font-bold uppercase tracking-wider text-stone-700 font-display">K-Nearest Neighbors (KNN)</span>
                        </div>
                        <div className="text-[11px] font-mono text-stone-500">
                          Testing accuracy: <span className="font-bold text-stone-800">{(modelMetrics.knnAcc * 100).toFixed(0)}%</span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                        <div className="md:col-span-4 bg-stone-50 p-2.5 rounded-lg border border-stone-200/60 text-center">
                          <span className="block text-[9px] text-stone-400 uppercase font-mono tracking-wider">Prediction</span>
                          <span className="text-sm font-semibold text-stone-800">{knnResult.prediction}</span>
                        </div>
                        <div className="md:col-span-8">
                          <div className="text-[10px] text-stone-400 uppercase tracking-widest font-mono mb-1">
                            Nearest matches in data-space (k={kParam}):
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {knnResult.neighbors.map((neighbor, index) => (
                              <span
                                key={index}
                                className={`px-2 py-1 rounded text-[10px] font-medium border ${
                                  neighbor.label === knnResult.prediction
                                    ? 'bg-blue-50/50 text-blue-700 border-blue-200'
                                    : 'bg-stone-50 text-stone-600 border-stone-200/60'
                                }`}
                              >
                                {neighbor.label} (d={neighbor.distance.toFixed(1)})
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Decision Tree Row */}
                    <div className="bg-white border border-stone-200/80 rounded-xl p-4 shadow-[0_1px_3px_rgba(0,0,0,0.01)]" id="card_tree_metric">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full bg-purple-500"></span>
                          <span className="text-xs font-bold uppercase tracking-wider text-stone-700 font-display">Decision Tree Traversal</span>
                        </div>
                        <div className="text-[11px] font-mono text-stone-500">
                          Testing accuracy: <span className="font-bold text-stone-800">{(modelMetrics.dtAcc * 100).toFixed(0)}%</span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-start">
                        {/* Static prediction box */}
                        <div className="md:col-span-4 bg-stone-50 p-2.5 rounded-lg border border-stone-200/60 text-center">
                          <span className="block text-[9px] text-stone-400 uppercase font-mono tracking-wider">Prediction</span>
                          <span className="text-sm font-semibold text-stone-800">{dtResult}</span>
                        </div>

                        {/* Interactive conditional splits listing */}
                        <div className="md:col-span-8">
                          <span className="block text-[9px] text-stone-400 uppercase tracking-widest font-mono mb-2">Live Node Split Journey:</span>
                          <div className="space-y-1.5 text-xs text-stone-600 PL-3 border-l border-stone-200">
                            {dtDecisionPathSteps.path.map((step, idx) => (
                              <div key={idx} className="flex flex-col">
                                <div className="flex items-center gap-1.5 text-stone-700 font-medium">
                                  <ChevronRight className="w-3 h-3 text-stone-400" />
                                  <span>Split {idx + 1}: {step.featureLabel}</span>
                                </div>
                                <span className="text-[10px] text-stone-400 ml-4 font-mono">
                                  ({step.currentVal}) {step.direction === 'Left' ? '<' : '>='} Threshold ({step.threshold.toFixed(1)}) → branch {step.direction}
                                </span>
                              </div>
                            ))}
                            <div className="text-[10px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded inline-block font-mono border border-emerald-100">
                              Terminal node reached → Class: {dtResult}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Random Forest Row */}
                    <div className="bg-white border border-stone-200/80 rounded-xl p-4 shadow-[0_1px_3px_rgba(0,0,0,0.01)]" id="card_rf_forest_metric">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full bg-pink-500"></span>
                          <span className="text-xs font-bold uppercase tracking-wider text-stone-700 font-display">Random Forest Vote Breakdown</span>
                        </div>
                        <div className="text-[11px] font-mono text-stone-500">
                          Testing accuracy: <span className="font-bold text-stone-800">{(modelMetrics.rfAcc * 100).toFixed(0)}%</span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                        <div className="md:col-span-4 bg-stone-50 p-2.5 rounded-lg border border-stone-200/60 text-center">
                          <span className="block text-[9px] text-stone-400 uppercase font-mono tracking-wider">Forest Choice</span>
                          <span className="text-sm font-semibold text-stone-800">{rfResult.prediction}</span>
                        </div>

                        {/* Visual Vote distribution bars */}
                        <div className="md:col-span-8 space-y-2">
                          {rfResult.voteBreakdown.slice(0, 3).map((vote) => (
                            <div key={vote.crop} className="space-y-0.5">
                              <div className="flex justify-between items-center text-[10px] font-mono text-stone-500">
                                <span className="font-semibold text-stone-700">{vote.crop}</span>
                                <span>{vote.votes} of {numTrees} trees ({vote.percentage}%)</span>
                              </div>
                              <div className="w-full bg-stone-100 h-1 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-pink-500 rounded-full transition-all duration-300"
                                  style={{ width: `${vote.percentage}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                  </div>
                </section>
              </motion.div>
            )}

            {activeTab === 'dataset' && (
              <motion.div
                key="tab_dataset_view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="bg-white border border-stone-200/80 rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)]"
                id="dataset_panel"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-stone-100 pb-4 mb-4">
                  <div>
                    <h3 className="text-base font-bold text-stone-900 font-display">Soil Baseline Reference Samples</h3>
                    <p className="text-xs text-stone-500 mt-0.5">Static reference dataset depicting optimal configurations for 10 distinct Indian crop profiles.</p>
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-stone-500 font-mono">Category class:</span>
                    <select
                      value={tableFilterCrop}
                      onChange={(e) => setTableFilterCrop(e.target.value)}
                      className="bg-stone-50 border border-stone-300 rounded px-2.5 py-1.5 focus:outline-none focus:border-stone-500 font-medium"
                      id="crop_table_filter"
                    >
                      <option value="All">All Crops ({dataset.length} profiles)</option>
                      {AVAILABLE_CROPS.map((crop) => (
                        <option key={crop} value={crop}>{crop}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Grid Table */}
                <div className="overflow-x-auto rounded-lg border border-stone-200/80 max-h-[500px]">
                  <table className="w-full text-xs text-left text-stone-600">
                    <thead className="bg-stone-50 text-stone-500 uppercase font-mono tracking-wider border-b border-stone-200/80 sticky top-0 bg-opacity-95 backdrop-blur-sm">
                      <tr>
                        <th className="px-4 py-3">#</th>
                        <th className="px-4 py-3">Nitrogen (N)</th>
                        <th className="px-4 py-3">Phosphorus (P)</th>
                        <th className="px-4 py-3">Potassium (K)</th>
                        <th className="px-4 py-3">Temperature</th>
                        <th className="px-4 py-3">pH Value</th>
                        <th className="px-4 py-3">Rainfall</th>
                        <th className="px-4 py-3 text-emerald-800">Class Label</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone-100">
                      {filteredRecords.map((item, index) => (
                        <tr key={index} className="hover:bg-stone-50/50 transition">
                          <td className="px-4 py-2.5 font-mono text-stone-400">{index + 1}</td>
                          <td className="px-4 py-2.5 font-mono">{item.n} mg/kg</td>
                          <td className="px-4 py-2.5 font-mono">{item.p} mg/kg</td>
                          <td className="px-4 py-2.5 font-mono">{item.k} mg/kg</td>
                          <td className="px-4 py-2.5 font-mono">{item.temperature}°C</td>
                          <td className="px-4 py-2.5 font-mono">{item.ph}</td>
                          <td className="px-4 py-2.5 font-mono">{item.rainfall} mm</td>
                          <td className="px-4 py-2.5 font-semibold text-emerald-700">{item.label}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {activeTab === 'about' && (
              <motion.div
                key="tab_about_view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-1 md:grid-cols-12 gap-6"
                id="about_panel"
              >
                <div className="md:col-span-8 bg-white border border-stone-200/80 rounded-xl p-6 space-y-6 shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
                  <div>
                    <h3 className="text-lg font-bold text-stone-900 font-display">Algorithmic Foundations of Crop Diagnostics</h3>
                    <p className="text-xs text-stone-500 mt-0.5">A look behind the technical mechanics of standard machine learning methodologies used in agricultural science.</p>
                  </div>

                  <div className="space-y-4 text-xs text-stone-600 leading-relaxed">
                    <div className="p-4 bg-stone-50 rounded-lg space-y-1.5 border border-stone-200/60">
                      <h4 className="font-bold text-stone-800 uppercase tracking-wide">1. K-Nearest Neighbors Classifier (Distance-Based)</h4>
                      <p>
                        KNN is a non-parametric metric classifier. It calculates the Euclidean distance between a target environmental vector and all baseline reference specimens:
                      </p>
                      <pre className="p-2 bg-stone-900 text-stone-300 font-mono rounded text-[10px] overflow-x-auto mt-1">
                        Distance = sqrt( sum( (User_i - Class_i)^2 ) )
                      </pre>
                      <p>
                        Features are scaled automatically first since distance metrics are highly sensitive to raw dimensions (e.g., rainfall millimeters vs. Soil pH scales). Majority class voting across K neighbors decides suitability.
                      </p>
                    </div>

                    <div className="p-4 bg-stone-50 rounded-lg space-y-1.5 border border-stone-200/60">
                      <h4 className="font-bold text-stone-800 uppercase tracking-wide">2. Decision Tree Classifier (Gini Impurity Splits)</h4>
                      <p>
                        The tree splits database datasets sequentially to minimize mathematical impurity, evaluated via Gini metrics. Traversal flows down left/right decision split criteria until a leaf node is isolated:
                      </p>
                      <pre className="p-2 bg-stone-900 text-stone-300 font-mono rounded text-[10px] overflow-x-auto mt-1">
                        Gini = 1 - sum( p_cls^2 )
                      </pre>
                      <p>
                        Each decision point partitions the multi-dimensional dataset relative to features (such as Nitrogen thresholds or ambient temperatures), mapping the trajectory directly to the terminal prediction.
                      </p>
                    </div>

                    <div className="p-4 bg-stone-50 rounded-lg space-y-1.5 border border-stone-200/60">
                      <h4 className="font-bold text-stone-800 uppercase tracking-wide">3. Random Forest (Ensemble Bootstrapping)</h4>
                      <p>
                        Random Forest trains an array of random trees in parallel over custom-sampled subsets of features and points (bagging). The aggregation (ensemble) decreases variance and prevents single-tree overfitting, outputting a robust consensus.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Quick details checklist */}
                <div className="md:col-span-4 bg-white border border-stone-200/80 rounded-xl p-5 space-y-4 shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-stone-800 font-display">Indian Soil Matrix Criteria</h4>
                  <p className="text-xs text-stone-500 leading-relaxed">
                    Indian agronomy is categorized into unique zones where specific macro-nutrient balances are key:
                  </p>
                  
                  <div className="space-y-3 text-xs">
                    <div className="flex gap-2.5 items-start">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5"></span>
                      <div>
                        <strong className="text-stone-700 block text-[11px]">Nitrogen (N)</strong>
                        <span className="text-[10px] text-stone-400">Promotes vegetative development and grass growth. Highly demanded by crops like Corn and Cotton.</span>
                      </div>
                    </div>
                    <div className="flex gap-2.5 items-start">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5"></span>
                      <div>
                        <strong className="text-stone-700 block text-[11px]">Phosphorus (P)</strong>
                        <span className="text-[10px] text-stone-400">Encourages root systems and initial seeding survival. Fruit soils require high levels.</span>
                      </div>
                    </div>
                    <div className="flex gap-2.5 items-start">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5"></span>
                      <div>
                        <strong className="text-stone-700 block text-[11px]">Potassium (K)</strong>
                        <span className="text-[10px] text-stone-400">Enables cellular wall moisture control and dynamic weather tolerance. Apples and Grapes are Potash-hungry.</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
      </main>

      {/* Thin elegant bottom status rail */}
      <footer className="bg-white border-t border-stone-200/80 px-6 py-3.5 mt-auto text-center text-[10px] font-mono text-stone-400 shrink-0" id="app_footer_rail">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-2">
          <span>Crop Recommendation Engine • Premium Minimal Layout</span>
          <span className="text-[10px] text-stone-400">Clean Local Storage • Pure Client-Side Inference Engines</span>
        </div>
      </footer>
    </div>
  );
}
