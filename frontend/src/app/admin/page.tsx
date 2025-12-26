'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Users, FileText, BarChart3, Shield, Activity, TrendingUp, Clock } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import { useAuthStore } from '@/lib/store';
import { adminAPI, authAPI } from '@/lib/api';

interface Stats {
  total_users: number;
  active_users: number;
  total_patients: number;
  total_documents: number;
  total_analyses: number;
  recent_analyses_7d: number;
  new_users_7d: number;
}

export default function AdminDashboard() {
  const router = useRouter();
  const { token, user } = useAuthStore();
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }
    checkAdminAndFetch();
  }, [token, router]);

  const checkAdminAndFetch = async () => {
    try {
      const me = await authAPI.me();
      if (me.role !== 'admin') {
        router.push('/dashboard');
        return;
      }
      const data = await adminAPI.getStats();
      setStats(data);
    } catch (err) {
      setError('Accès refusé ou erreur de chargement');
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) return null;

  const statCards = stats ? [
    { label: 'Utilisateurs', value: stats.total_users, icon: Users, color: 'from-blue-500 to-cyan-500', sub: `${stats.active_users} actifs` },
    { label: 'Patients', value: stats.total_patients, icon: FileText, color: 'from-emerald-500 to-teal-500', sub: 'Total enregistrés' },
    { label: 'Documents', value: stats.total_documents, icon: FileText, color: 'from-purple-500 to-pink-500', sub: 'Uploadés' },
    { label: 'Analyses', value: stats.total_analyses, icon: BarChart3, color: 'from-orange-500 to-red-500', sub: `${stats.recent_analyses_7d} cette semaine` },
  ] : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-12 h-12 bg-gradient-to-br from-rose-500/20 to-orange-500/20 rounded-xl flex items-center justify-center border border-rose-500/20">
            <Shield className="h-6 w-6 text-rose-400" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Administration</h1>
            <p className="text-slate-400">Gérez la plateforme et les utilisateurs</p>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        ) : (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {statCards.map((stat, i) => (
                <div key={i} className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`w-12 h-12 bg-gradient-to-br ${stat.color} rounded-xl flex items-center justify-center opacity-80`}>
                      <stat.icon className="h-6 w-6 text-white" />
                    </div>
                    <TrendingUp className="h-5 w-5 text-emerald-400" />
                  </div>
                  <p className="text-3xl font-bold text-white">{stat.value}</p>
                  <p className="text-slate-400 text-sm">{stat.label}</p>
                  <p className="text-slate-500 text-xs mt-1">{stat.sub}</p>
                </div>
              ))}
            </div>

            {/* Quick Links */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Link href="/admin/users">
                <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6 hover:bg-slate-800/50 hover:border-slate-600/50 transition-all cursor-pointer group">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-blue-500/10 rounded-xl flex items-center justify-center border border-blue-500/20 group-hover:scale-110 transition-transform">
                      <Users className="h-7 w-7 text-blue-400" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-white">Gestion des Utilisateurs</h3>
                      <p className="text-slate-400">Gérer les comptes, rôles et permissions</p>
                    </div>
                  </div>
                </div>
              </Link>

              <Link href="/admin/logs">
                <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6 hover:bg-slate-800/50 hover:border-slate-600/50 transition-all cursor-pointer group">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-purple-500/10 rounded-xl flex items-center justify-center border border-purple-500/20 group-hover:scale-110 transition-transform">
                      <Activity className="h-7 w-7 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-white">Logs & Traçabilité</h3>
                      <p className="text-slate-400">Consulter l'historique des actions</p>
                    </div>
                  </div>
                </div>
              </Link>
            </div>

            {/* Recent Activity */}
            {stats && stats.new_users_7d > 0 && (
              <div className="mt-8 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Clock className="h-5 w-5 text-amber-400" />
                  <h3 className="text-lg font-semibold text-white">Activité Récente (7 jours)</h3>
                </div>
                <div className="flex gap-8 text-slate-400">
                  <div><span className="text-white font-bold">{stats.new_users_7d}</span> nouveaux utilisateurs</div>
                  <div><span className="text-white font-bold">{stats.recent_analyses_7d}</span> analyses effectuées</div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

