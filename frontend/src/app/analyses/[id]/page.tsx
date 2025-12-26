'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { ArrowLeft, AlertTriangle, Brain, Clock, BarChart3, Sparkles } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import Disclaimer from '@/components/layout/Disclaimer';
import { useAuthStore } from '@/lib/store';
import { predictionAPI } from '@/lib/api';

interface FeatureImportance {
  feature_name: string;
  importance_score: number;
  contribution: string;
}

interface Analysis {
  id: string;
  patient_id: string;
  patient_code?: string;
  model_type: string;
  risk_score: number;
  risk_level: string;
  risk_percentage: number;
  feature_importance: FeatureImportance[];
  model_version: string;
  created_at: string;
  disclaimer: string;
}

export default function AnalysisDetailPage() {
  const router = useRouter();
  const params = useParams();
  const analysisId = params.id as string;
  const { token } = useAuthStore();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }
    fetchAnalysis();
  }, [token, router, analysisId]);

  const fetchAnalysis = async () => {
    try {
      const data = await predictionAPI.getAnalysis(analysisId);
      setAnalysis(data);
    } catch (error) {
      console.error('Error fetching analysis:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) return null;

  const getRiskGradient = (level: string) => {
    switch (level) {
      case 'low': return 'from-emerald-500 to-teal-500';
      case 'moderate': return 'from-amber-500 to-yellow-500';
      case 'high': return 'from-orange-500 to-red-500';
      case 'very_high': return 'from-rose-500 to-pink-500';
      default: return 'from-slate-500 to-slate-600';
    }
  };

  const getRiskLabel = (level: string) => {
    switch (level) {
      case 'low': return 'Risque Faible';
      case 'moderate': return 'Risque Modéré';
      case 'high': return 'Risque Élevé';
      case 'very_high': return 'Risque Très Élevé';
      default: return level;
    }
  };

  const getRiskTextColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-emerald-400';
      case 'moderate': return 'text-amber-400';
      case 'high': return 'text-orange-400';
      case 'very_high': return 'text-rose-400';
      default: return 'text-slate-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Disclaimer />

        <button onClick={() => router.push('/analyses')} className="flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4 mr-2" /> Retour aux analyses
        </button>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        ) : !analysis ? (
          <div className="text-center py-12 text-slate-400">Analyse non trouvée</div>
        ) : (
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center border border-purple-500/20">
                <Sparkles className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Résultat de Prédiction</h1>
                <div className="flex items-center gap-3 text-sm text-slate-400 mt-1">
                  <span className="flex items-center gap-1">
                    <Brain className="h-3 w-3" />
                    {analysis.model_type === 'breast_cancer' ? 'Cancer du sein' : analysis.model_type}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {new Date(analysis.created_at).toLocaleDateString('fr-FR', {
                      day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit'
                    })}
                  </span>
                </div>
              </div>
            </div>

            {/* Risk Score Card */}
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-8">
              <div className="text-center">
                <div className={`inline-flex items-center justify-center w-40 h-40 rounded-full bg-gradient-to-br ${getRiskGradient(analysis.risk_level)} p-1 mb-6`}>
                  <div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center">
                    <div className="text-center">
                      <p className="text-5xl font-bold text-white">
                        {Math.round(analysis.risk_percentage)}%
                      </p>
                    </div>
                  </div>
                </div>
                <h2 className={`text-2xl font-bold ${getRiskTextColor(analysis.risk_level)}`}>
                  {getRiskLabel(analysis.risk_level)}
                </h2>
                <p className="text-slate-400 mt-2">Score de malignité prédit par le modèle IA</p>

                {/* Progress bar */}
                <div className="max-w-md mx-auto mt-8">
                  <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${getRiskGradient(analysis.risk_level)} transition-all duration-1000`}
                      style={{ width: `${analysis.risk_percentage}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-slate-500 mt-2">
                    <span>0%</span>
                    <span>25%</span>
                    <span>50%</span>
                    <span>75%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Feature Importance */}
            {analysis.feature_importance && analysis.feature_importance.length > 0 && (
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
                <div className="p-5 border-b border-slate-700/50">
                  <div className="flex items-center gap-3">
                    <BarChart3 className="h-5 w-5 text-blue-400" />
                    <div>
                      <h2 className="text-lg font-semibold text-white">Facteurs les plus importants</h2>
                      <p className="text-sm text-slate-400">Explicabilité du modèle (XAI)</p>
                    </div>
                  </div>
                </div>
                <div className="p-5 space-y-4">
                  {analysis.feature_importance.map((feat, index) => (
                    <div key={index} className="flex items-center gap-4">
                      <span className="font-medium text-white w-48 truncate">{feat.feature_name}</span>
                      <div className="flex-1 flex items-center gap-4">
                        <div className="flex-1 h-2 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
                            style={{ width: `${Math.round(feat.importance_score * 100)}%` }}
                          />
                        </div>
                        <span className="text-sm text-slate-400 w-12 text-right">
                          {Math.round(feat.importance_score * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Disclaimer Box */}
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <AlertTriangle className="h-5 w-5 text-amber-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-amber-400 mb-2">⚠️ Avertissement Important</h3>
                  <p className="text-slate-300">{analysis.disclaimer || "Ce résultat est fourni à titre indicatif uniquement et ne constitue pas un diagnostic médical. Consultez un professionnel de santé qualifié."}</p>
                  <p className="text-slate-500 text-sm mt-3">
                    Version du modèle: {analysis.model_version || 'v1.0.0'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

