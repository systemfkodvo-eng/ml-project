'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Activity, Search, Clock, User, Filter } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import { useAuthStore } from '@/lib/store';
import { adminAPI, authAPI } from '@/lib/api';

interface AuditLog {
  id: string;
  user_id: string;
  user_email: string;
  user_name: string;
  action: string;
  details: string;
  ip_address?: string;
  created_at: string;
}

export default function AdminLogsPage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [actionFilter, setActionFilter] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!token) { router.push('/login'); return; }
    checkAdminAndFetch();
  }, [token, router]);

  const checkAdminAndFetch = async () => {
    try {
      const me = await authAPI.me();
      if (me.role !== 'admin') { router.push('/dashboard'); return; }
      await fetchLogs();
    } catch { router.push('/dashboard'); }
  };

  const fetchLogs = async (action?: string) => {
    setIsLoading(true);
    try {
      const data = await adminAPI.listLogs({ action: action || undefined, limit: 200 });
      setLogs(Array.isArray(data) ? data : []);
    } catch (e) { console.error(e); }
    finally { setIsLoading(false); }
  };

  const handleFilter = (e: React.FormEvent) => { e.preventDefault(); fetchLogs(actionFilter); };

  const getActionBadge = (action: string) => {
    const colors: Record<string, string> = {
      login: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      logout: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
      create_patient: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      upload_document: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
      run_prediction: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      toggle_user_active: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
      update_user_role: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
      delete_user: 'bg-red-500/10 text-red-400 border-red-500/20',
    };
    const color = colors[action] || 'bg-slate-500/10 text-slate-400 border-slate-500/20';
    return <span className={`px-2 py-1 ${color} border rounded-full text-xs`}>{action}</span>;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  if (!token) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <button onClick={() => router.push('/admin')} className="flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4 mr-2" /> Retour à l'administration
        </button>

        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center border border-purple-500/20">
            <Activity className="h-5 w-5 text-purple-400" />
          </div>
          <h1 className="text-2xl font-bold text-white">Logs & Traçabilité</h1>
        </div>

        <form onSubmit={handleFilter} className="mb-6 flex gap-4">
          <div className="flex-1 relative">
            <Filter className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
            <input value={actionFilter} onChange={(e) => setActionFilter(e.target.value)} placeholder="Filtrer par action (ex: login, create_patient)..."
              className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50" />
          </div>
          <button type="submit" className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors">Filtrer</button>
          {actionFilter && <button type="button" onClick={() => { setActionFilter(''); fetchLogs(); }} className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-colors">Réinitialiser</button>}
        </form>

        {isLoading ? (
          <div className="flex items-center justify-center py-12"><div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin"></div></div>
        ) : logs.length === 0 ? (
          <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-12 text-center">
            <Activity className="h-12 w-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">Aucun log trouvé</p>
            <p className="text-slate-500 text-sm mt-2">Les actions des utilisateurs apparaîtront ici</p>
          </div>
        ) : (
          <div className="space-y-3">
            {logs.map((log) => (
              <div key={log.id} className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4 hover:bg-slate-800/50 transition-colors">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-slate-700/50 rounded-full flex items-center justify-center flex-shrink-0">
                      <User className="h-5 w-5 text-slate-400" />
                    </div>
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <span className="font-medium text-white">{log.user_name || 'Utilisateur'}</span>
                        {getActionBadge(log.action)}
                      </div>
                      <p className="text-slate-400 text-sm">{log.details || 'Aucun détail'}</p>
                      <p className="text-slate-500 text-xs mt-1">{log.user_email}</p>
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className="flex items-center gap-1 text-slate-500 text-sm">
                      <Clock className="h-4 w-4" />
                      {formatDate(log.created_at)}
                    </div>
                    {log.ip_address && <p className="text-slate-600 text-xs mt-1">IP: {log.ip_address}</p>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

