'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { BarChart3, TrendingUp, AlertTriangle, Brain, Clock } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import Disclaimer from '@/components/layout/Disclaimer';
import { useAuthStore } from '@/lib/store';
import { predictionAPI } from '@/lib/api';

interface Analysis {
  id: string;
  patient_id: string;
  patient_code?: string;
  model_type: string;
  risk_score: number;
  risk_level: string;
  risk_percentage: number;
  created_at: string;
}

export default function AnalysesPage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }
    fetchAnalyses();
  }, [token, router]);

  const fetchAnalyses = async () => {
    try {
      const data = await predictionAPI.listAnalyses();
      setAnalyses(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error('Error fetching analyses:', error);
      setAnalyses([]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) return null;

  const getRiskBadge = (level: string) => {
    switch (level) {
      case 'low': return <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-sm font-medium">Faible</span>;
      case 'moderate': return <span className="px-3 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-full text-sm font-medium">Modéré</span>;
      case 'high': return <span className="px-3 py-1 bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded-full text-sm font-medium">Élevé</span>;
      case 'very_high': return <span className="px-3 py-1 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded-full text-sm font-medium">Très Élevé</span>;
      default: return <span className="px-3 py-1 bg-slate-500/10 text-slate-400 rounded-full text-sm">{level}</span>;
    }
  };

  const getRiskIcon = (level: string) => {
    if (level === 'very_high' || level === 'high') {
      return <AlertTriangle className="h-6 w-6 text-rose-400" />;
    }
    return <TrendingUp className="h-6 w-6 text-emerald-400" />;
  };

  const getRiskIconBg = (level: string) => {
    switch (level) {
      case 'low': return 'bg-emerald-500/10 border-emerald-500/20';
      case 'moderate': return 'bg-amber-500/10 border-amber-500/20';
      case 'high': return 'bg-orange-500/10 border-orange-500/20';
      case 'very_high': return 'bg-rose-500/10 border-rose-500/20';
      default: return 'bg-slate-500/10 border-slate-500/20';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Disclaimer />

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center border border-purple-500/20">
              <Brain className="h-5 w-5 text-purple-400" />
            </div>
            <h1 className="text-3xl font-bold text-white">Historique des Analyses</h1>
          </div>
          <p className="text-slate-400 mt-2">Consultez toutes les prédictions de risque effectuées</p>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        ) : analyses.length === 0 ? (
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12 text-center">
            <div className="w-16 h-16 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="h-8 w-8 text-slate-400" />
            </div>
            <p className="text-slate-400 mb-2">Aucune analyse effectuée</p>
            <p className="text-sm text-slate-500">
              Uploadez un document, validez les données extraites, puis lancez une prédiction
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {analyses.map((analysis) => (
              <Link key={analysis.id} href={`/analyses/${analysis.id}`}>
                <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-5 hover:bg-slate-800/50 hover:border-slate-600/50 transition-all cursor-pointer">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${getRiskIconBg(analysis.risk_level)}`}>
                        {getRiskIcon(analysis.risk_level)}
                      </div>
                      <div>
                        <p className="font-semibold text-white text-lg">
                          {analysis.patient_code || `Patient ${analysis.patient_id.slice(0, 8)}...`}
                        </p>
                        <div className="flex items-center gap-3 text-sm text-slate-400 mt-1">
                          <span className="flex items-center gap-1">
                            <Brain className="h-3 w-3" />
                            {analysis.model_type === 'breast_cancer' ? 'Cancer du sein' : analysis.model_type}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(analysis.created_at).toLocaleDateString('fr-FR', {
                              day: 'numeric',
                              month: 'long',
                              year: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className="text-3xl font-bold text-white">{Math.round(analysis.risk_percentage)}%</p>
                        <p className="text-sm text-slate-500">Score de risque</p>
                      </div>
                      {getRiskBadge(analysis.risk_level)}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

