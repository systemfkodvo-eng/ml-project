'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Plus, Search, User, FileText, BarChart, X, Trash2 } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import { useAuthStore } from '@/lib/store';
import { patientsAPI } from '@/lib/api';

interface Patient {
  id: string;
  patient_code: string;
  gender?: string;
  date_of_birth?: string;
  document_count?: number;
  analysis_count?: number;
  created_at: string;
}

export default function PatientsPage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [search, setSearch] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [showNewPatient, setShowNewPatient] = useState(false);
  const [newPatient, setNewPatient] = useState({ patient_code: '', gender: '', notes: '' });
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }
    fetchPatients();
  }, [token, router]);

  const fetchPatients = async (searchTerm?: string) => {
    setIsLoading(true);
    try {
      const data = await patientsAPI.list({ search: searchTerm });
      setPatients(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error('Error fetching patients:', error);
      setPatients([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchPatients(search);
  };

  const handleCreatePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setError('');
    try {
	      // Clean payload before sending to API to avoid validation errors
	      const payload: {
	        patient_code: string;
	        gender?: string;
	        notes?: string;
	      } = {
	        patient_code: newPatient.patient_code,
	      };

	      // Only send gender if a valid value is selected ("male" | "female")
	      if (newPatient.gender) {
	        payload.gender = newPatient.gender;
	      }

	      // Only send notes if non-empty
	      if (newPatient.notes && newPatient.notes.trim().length > 0) {
	        payload.notes = newPatient.notes.trim();
	      }

	      await patientsAPI.create(payload);
      setShowNewPatient(false);
      setNewPatient({ patient_code: '', gender: '', notes: '' });
      fetchPatients();
    } catch (err: unknown) {
	      const error = err as { response?: { data?: { detail?: unknown } } };
	      const detail = error.response?.data?.detail;
	      if (typeof detail === 'string') {
	        setError(detail);
	      } else if (Array.isArray(detail)) {
	        // FastAPI 422 validation error: join messages
	        const msgs = (detail as Array<{ msg?: string }>).map((d) => d.msg).filter(Boolean);
	        setError(msgs.join('\n') || 'Erreur lors de la création');
	      } else {
	        setError('Erreur lors de la création');
	      }
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm('Supprimer ce patient et toutes ses données ?')) {
      try {
        await patientsAPI.delete(id);
        fetchPatients();
      } catch (error) {
        console.error('Error deleting patient:', error);
      }
    }
  };

  if (!token) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Dossiers Patients</h1>
            <p className="text-slate-400 mt-1">Gérez vos dossiers patients pseudonymisés</p>
          </div>
          <button
            onClick={() => setShowNewPatient(true)}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all"
          >
            <Plus className="h-5 w-5" /> Nouveau Patient
          </button>
        </div>

        {/* Search */}
        <form onSubmit={handleSearch} className="mb-8 flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Rechercher par code patient..."
              className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            />
          </div>
          <button type="submit" className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors">
            Rechercher
          </button>
        </form>

        {/* New Patient Modal */}
        {showNewPatient && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="w-full max-w-md bg-slate-800/90 backdrop-blur-xl rounded-2xl border border-slate-700/50 p-6 shadow-2xl">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold text-white">Nouveau Patient</h2>
                <button onClick={() => setShowNewPatient(false)} className="text-slate-400 hover:text-white">
                  <X className="h-6 w-6" />
                </button>
              </div>

              {error && (
                <div className="mb-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg text-sm">
                  {error}
                </div>
              )}

              <form onSubmit={handleCreatePatient} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Code Patient (pseudonymisé)</label>
                  <input
                    value={newPatient.patient_code}
                    onChange={(e) => setNewPatient({ ...newPatient, patient_code: e.target.value })}
                    placeholder="PAT-2024-001"
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-600/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Genre</label>
                  <select
                    value={newPatient.gender}
                    onChange={(e) => setNewPatient({ ...newPatient, gender: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-600/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  >
                    <option value="" className="bg-slate-900">Sélectionner</option>
                    <option value="male" className="bg-slate-900">Homme</option>
                    <option value="female" className="bg-slate-900">Femme</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Notes (optionnel)</label>
                  <textarea
                    value={newPatient.notes}
                    onChange={(e) => setNewPatient({ ...newPatient, notes: e.target.value })}
                    placeholder="Notes additionnelles..."
                    rows={3}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-600/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-none"
                  />
                </div>
                <div className="flex gap-4 pt-2">
                  <button
                    type="button"
                    onClick={() => setShowNewPatient(false)}
                    className="flex-1 py-3 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-xl transition-colors"
                  >
                    Annuler
                  </button>
                  <button
                    type="submit"
                    disabled={creating}
                    className="flex-1 py-3 px-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-semibold rounded-xl disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {creating ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> : 'Créer'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Patients List */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        ) : patients.length === 0 ? (
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12 text-center">
            <div className="w-16 h-16 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
              <User className="h-8 w-8 text-slate-400" />
            </div>
            <p className="text-slate-400 mb-6">Aucun patient trouvé</p>
            <button
              onClick={() => setShowNewPatient(true)}
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25"
            >
              Ajouter un patient
            </button>
          </div>
        ) : (
          <div className="grid gap-4">
            {patients.map((patient) => (
              <Link key={patient.id} href={`/patients/${patient.id}`}>
                <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-5 hover:bg-slate-800/50 hover:border-slate-600/50 transition-all cursor-pointer group">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center border border-blue-500/20">
                        <User className="h-6 w-6 text-blue-400" />
                      </div>
                      <div>
                        <p className="font-semibold text-white text-lg">{patient.patient_code}</p>
                        <p className="text-sm text-slate-400">
                          {patient.gender === 'male' ? 'Homme' : patient.gender === 'female' ? 'Femme' : 'Non spécifié'} •
                          Créé le {new Date(patient.created_at).toLocaleDateString('fr-FR')}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-6">
                      <div className="flex items-center gap-2 text-slate-400">
                        <FileText className="h-4 w-4" />
                        <span>{patient.document_count || 0} docs</span>
                      </div>
                      <div className="flex items-center gap-2 text-slate-400">
                        <BarChart className="h-4 w-4" />
                        <span>{patient.analysis_count || 0} analyses</span>
                      </div>
                      <button
                        onClick={(e) => handleDelete(patient.id, e)}
                        className="p-2 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                      >
                        <Trash2 className="h-5 w-5" />
                      </button>
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

