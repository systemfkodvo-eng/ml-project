'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Users, Search, Shield, UserCheck, UserX, Trash2, Edit } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import { useAuthStore } from '@/lib/store';
import { adminAPI, authAPI } from '@/lib/api';

interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
  specialty?: string;
  institution?: string;
  is_active: boolean;
  created_at: string;
}

export default function AdminUsersPage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [users, setUsers] = useState<User[]>([]);
  const [search, setSearch] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [editingRole, setEditingRole] = useState<string | null>(null);

  useEffect(() => {
    if (!token) { router.push('/login'); return; }
    checkAdminAndFetch();
  }, [token, router]);

  const checkAdminAndFetch = async () => {
    try {
      const me = await authAPI.me();
      if (me.role !== 'admin') { router.push('/dashboard'); return; }
      await fetchUsers();
    } catch { router.push('/dashboard'); }
  };

  const fetchUsers = async (searchTerm?: string) => {
    setIsLoading(true);
    try {
      const data = await adminAPI.listUsers({ search: searchTerm });
      setUsers(Array.isArray(data) ? data : []);
    } catch (e) { console.error(e); }
    finally { setIsLoading(false); }
  };

  const handleSearch = (e: React.FormEvent) => { e.preventDefault(); fetchUsers(search); };

  const handleToggleActive = async (userId: string) => {
    try {
      await adminAPI.toggleUserActive(userId);
      fetchUsers(search);
    } catch (e) { console.error(e); }
  };

  const handleUpdateRole = async (userId: string, role: string) => {
    try {
      await adminAPI.updateUserRole(userId, role);
      setEditingRole(null);
      fetchUsers(search);
    } catch (e) { console.error(e); }
  };

  const handleDelete = async (userId: string) => {
    if (!confirm('Supprimer cet utilisateur définitivement ?')) return;
    try {
      await adminAPI.deleteUser(userId);
      fetchUsers(search);
    } catch (e) { console.error(e); }
  };

  const getRoleBadge = (role: string) => {
    switch (role) {
      case 'admin': return <span className="px-2 py-1 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded-full text-xs">Admin</span>;
      case 'practitioner': return <span className="px-2 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full text-xs">Praticien</span>;
      case 'researcher': return <span className="px-2 py-1 bg-purple-500/10 text-purple-400 border border-purple-500/20 rounded-full text-xs">Chercheur</span>;
      default: return <span className="px-2 py-1 bg-slate-500/10 text-slate-400 rounded-full text-xs">{role}</span>;
    }
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
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center border border-blue-500/20">
            <Users className="h-5 w-5 text-blue-400" />
          </div>
          <h1 className="text-2xl font-bold text-white">Gestion des Utilisateurs</h1>
        </div>

        <form onSubmit={handleSearch} className="mb-6 flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Rechercher par email ou nom..."
              className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50" />
          </div>
          <button type="submit" className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors">Rechercher</button>
        </form>

        {isLoading ? (
          <div className="flex items-center justify-center py-12"><div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div></div>
        ) : users.length === 0 ? (
          <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-12 text-center">
            <Users className="h-12 w-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">Aucun utilisateur trouvé</p>
          </div>
        ) : (
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-800/50">
                  <tr className="text-left text-slate-400 text-sm">
                    <th className="px-6 py-4">Utilisateur</th>
                    <th className="px-6 py-4">Rôle</th>
                    <th className="px-6 py-4">Statut</th>
                    <th className="px-6 py-4">Créé le</th>
                    <th className="px-6 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {users.map((user) => (
                    <tr key={user.id} className="hover:bg-slate-700/20">
                      <td className="px-6 py-4">
                        <div><p className="font-medium text-white">{user.full_name}</p><p className="text-sm text-slate-400">{user.email}</p></div>
                      </td>
                      <td className="px-6 py-4">
                        {editingRole === user.id ? (
                          <select value={user.role} onChange={(e) => handleUpdateRole(user.id, e.target.value)} onBlur={() => setEditingRole(null)}
                            className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-white text-sm">
                            <option value="admin">Admin</option>
                            <option value="practitioner">Praticien</option>
                            <option value="researcher">Chercheur</option>
                          </select>
                        ) : (
                          <button onClick={() => setEditingRole(user.id)} className="flex items-center gap-2">{getRoleBadge(user.role)}<Edit className="h-3 w-3 text-slate-500" /></button>
                        )}
                      </td>
                      <td className="px-6 py-4">
                        {user.is_active ? <span className="flex items-center gap-1 text-emerald-400 text-sm"><UserCheck className="h-4 w-4" />Actif</span>
                          : <span className="flex items-center gap-1 text-rose-400 text-sm"><UserX className="h-4 w-4" />Inactif</span>}
                      </td>
                      <td className="px-6 py-4 text-slate-400 text-sm">{user.created_at ? new Date(user.created_at).toLocaleDateString('fr-FR') : '-'}</td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button onClick={() => handleToggleActive(user.id)} className={`p-2 rounded-lg transition-colors ${user.is_active ? 'text-amber-400 hover:bg-amber-500/10' : 'text-emerald-400 hover:bg-emerald-500/10'}`}>
                            {user.is_active ? <UserX className="h-4 w-4" /> : <UserCheck className="h-4 w-4" />}
                          </button>
                          <button onClick={() => handleDelete(user.id)} className="p-2 text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"><Trash2 className="h-4 w-4" /></button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

