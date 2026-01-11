'use client';

import { useState } from 'react';
import { SearchBar } from '@/components/SearchBar';
import { StatusDisplay } from '@/components/StatusDisplay';
import { GameGrid } from '@/components/GameGrid';
import { GameModal } from '@/components/GameModal';
import { searchGames } from '@/lib/api';
import { SearchState, GameResult } from '@/lib/types';
import { cn } from '@/lib/utils'; // Make sure this import is correct

export default function Home() {
  const [state, setState] = useState<SearchState>({
    status: 'idle',
    query: '',
    interpretation: null,
    results: [],
    error: null,
  });

  const [selectedGame, setSelectedGame] = useState<GameResult | null>(null);

  const handleSearch = async (query: string) => {
    // Reset state for new search
    setState(prev => ({
      ...prev,
      status: 'interpreting',
      query,
      interpretation: null,
      results: [],
      error: null
    }));
    setSelectedGame(null);

    await searchGames(query, (update) => {
      setState(prev => ({ ...prev, ...update }));
    });
  };

  const isHasResults = state.results.length > 0;
  const isSearching = state.status === 'interpreting' || state.status === 'searching';

  return (
    <main className="min-h-screen flex flex-col p-4 md:p-8">

      {/* Hero / Header Section */}
      <div className={cn(
        "transition-all duration-500 ease-in-out flex flex-col items-center",
        isHasResults || isSearching ? "mt-4 mb-8" : "mt-[30vh]"
      )}>
        <h1 className="text-4xl md:text-6xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-br from-white to-steam-text mb-6">
          STEAM <span className="text-steam-light-blue">ORACLE</span>
        </h1>

        <SearchBar
          onSearch={handleSearch}
          isLoading={isSearching}
        />

        {state.error && (
          <div className="mt-4 p-4 bg-red-900/20 border border-red-500/50 rounded text-red-200">
            Error: {state.error}
          </div>
        )}
      </div>

      {/* Results Section */}
      <StatusDisplay
        status={state.status}
        interpretation={state.interpretation}
      />

      <GameGrid
        results={state.results}
        isLoading={isSearching}
        onSelectGame={setSelectedGame}
      />

      {/* Detail Modal */}
      {selectedGame && (
        <GameModal
          game={selectedGame}
          onClose={() => setSelectedGame(null)}
        />
      )}

    </main>
  );
}
