'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Users, FileText, Activity, TrendingUp, Plus, Brain, Shield, ArrowRight, Sparkles } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import Disclaimer from '@/components/layout/Disclaimer';
import { useAuthStore } from '@/lib/store';
import { patientsAPI, predictionAPI } from '@/lib/api';

interface DashboardStats {
  patients: number;
  analyses: number;
  highRisk: number;
}

export default function DashboardPage() {
  const router = useRouter();
  const { token, user } = useAuthStore();
  const [stats, setStats] = useState<DashboardStats>({ patients: 0, analyses: 0, highRisk: 0 });
  const [recentAnalyses, setRecentAnalyses] = useState<Array<{
    id: string;
    patient_id: string;
    risk_level: string;
    risk_percentage: number;
    created_at: string;
  }>>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }

    const fetchData = async () => {
      try {
        setError(null);
        const [patients, analyses] = await Promise.all([
          patientsAPI.list().catch(() => []),
          predictionAPI.listAnalyses({ limit: 5 }).catch(() => []),
        ]);

        setStats({
          patients: Array.isArray(patients) ? patients.length : 0,
          analyses: Array.isArray(analyses) ? analyses.length : 0,
          highRisk: Array.isArray(analyses) ? analyses.filter((a: { risk_level: string }) => ['high', 'very_high'].includes(a.risk_level)).length : 0,
        });
        setRecentAnalyses(Array.isArray(analyses) ? analyses : []);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Erreur de connexion au serveur');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [token, router]);

  if (!token) return null;

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'moderate': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'high': return 'bg-orange-500/10 text-orange-400 border-orange-500/20';
      case 'very_high': return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      default: return 'bg-slate-500/10 text-slate-400 border-slate-500/20';
    }
  };

  const getRiskLabel = (level: string) => {
    switch (level) {
      case 'low': return 'Faible';
      case 'moderate': return 'Modéré';
      case 'high': return 'Élevé';
      case 'very_high': return 'Très élevé';
      default: return level;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />

      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/2 -left-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute -bottom-40 right-1/3 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      <main className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Disclaimer />

        {/* Header */}
        <div className="mb-10">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500">
              <Sparkles className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              Bienvenue, {user?.full_name || 'Docteur'}
            </h1>
          </div>
          <p className="text-slate-400 ml-14">Tableau de bord IA pour l&apos;analyse prédictive de santé</p>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400">
            {error}
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 p-6 backdrop-blur-xl hover:border-blue-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium mb-1">Total Patients</p>
                <p className="text-4xl font-bold text-white">{isLoading ? '—' : stats.patients}</p>
                <p className="text-blue-400 text-sm mt-2 flex items-center gap-1">
                  <Plus className="h-3 w-3" /> Ajouter un patient
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-500/20">
                <Users className="h-8 w-8 text-blue-400" />
              </div>
            </div>
          </div>

          <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 p-6 backdrop-blur-xl hover:border-emerald-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-500/10">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-teal-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium mb-1">Analyses IA</p>
                <p className="text-4xl font-bold text-white">{isLoading ? '—' : stats.analyses}</p>
                <p className="text-emerald-400 text-sm mt-2 flex items-center gap-1">
                  <Brain className="h-3 w-3" /> Prédictions ML
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/20">
                <FileText className="h-8 w-8 text-emerald-400" />
              </div>
            </div>
          </div>

          <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 p-6 backdrop-blur-xl hover:border-rose-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-rose-500/10">
            <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-orange-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium mb-1">Risque Élevé</p>
                <p className="text-4xl font-bold text-white">{isLoading ? '—' : stats.highRisk}</p>
                <p className="text-rose-400 text-sm mt-2 flex items-center gap-1">
                  <Shield className="h-3 w-3" /> Surveillance active
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-gradient-to-br from-rose-500/20 to-orange-500/20 border border-rose-500/20">
                <Activity className="h-8 w-8 text-rose-400" />
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
          <button
            onClick={() => router.push('/patients')}
            className="group relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-600 to-cyan-600 p-6 text-left hover:shadow-lg hover:shadow-blue-500/25 transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-white mb-1">Nouveau Patient</h3>
                <p className="text-blue-100">Créer un dossier patient pseudonymisé</p>
              </div>
              <ArrowRight className="h-6 w-6 text-white group-hover:translate-x-1 transition-transform" />
            </div>
          </button>

          <button
            onClick={() => router.push('/analyses')}
            className="group relative overflow-hidden rounded-2xl bg-gradient-to-r from-purple-600 to-pink-600 p-6 text-left hover:shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-white mb-1">Voir Analyses</h3>
                <p className="text-purple-100">Consulter les prédictions IA</p>
              </div>
              <ArrowRight className="h-6 w-6 text-white group-hover:translate-x-1 transition-transform" />
            </div>
          </button>
        </div>

        {/* Recent Analyses */}
        <div className="rounded-2xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 backdrop-blur-xl overflow-hidden">
          <div className="p-6 border-b border-slate-700/50">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/20">
                <TrendingUp className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Analyses Récentes</h2>
                <p className="text-slate-400 text-sm">Les 5 dernières prédictions IA</p>
              </div>
            </div>
          </div>

          <div className="p-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
              </div>
            ) : recentAnalyses.length === 0 ? (
              <div className="text-center py-12">
                <Brain className="h-16 w-16 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400 text-lg">Aucune analyse pour le moment</p>
                <p className="text-slate-500 text-sm mt-1">Uploadez un document pour lancer une prédiction IA</p>
              </div>
            ) : (
              <div className="space-y-3">
                {recentAnalyses.map((analysis) => (
                  <div
                    key={analysis.id}
                    className="group flex items-center justify-between p-4 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-slate-600 transition-all cursor-pointer"
                    onClick={() => router.push(`/analyses/${analysis.id}`)}
                  >
                    <div className="flex items-center gap-4">
                      <div className="p-2 rounded-lg bg-slate-700/50">
                        <Activity className="h-5 w-5 text-slate-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white">Patient: {analysis.patient_id.slice(0, 8)}...</p>
                        <p className="text-sm text-slate-400">
                          {new Date(analysis.created_at).toLocaleDateString('fr-FR', {
                            day: 'numeric',
                            month: 'short',
                            year: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="text-2xl font-bold text-white">{analysis.risk_percentage}%</p>
                        <p className="text-xs text-slate-400">Score de risque</p>
                      </div>
                      <span className={`px-3 py-1.5 rounded-lg text-sm font-medium border ${getRiskColor(analysis.risk_level)}`}>
                        {getRiskLabel(analysis.risk_level)}
                      </span>
                      <ArrowRight className="h-5 w-5 text-slate-500 group-hover:text-white group-hover:translate-x-1 transition-all" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

