import { Search } from 'lucide-react';
import { useState, FormEvent } from 'react';
import { cn } from '@/lib/utils'; // I need to create utils.ts first actually, or inline clsx

interface SearchBarProps {
    onSearch: (query: string) => void;
    isLoading: boolean;
    className?: string;
}

export function SearchBar({ onSearch, isLoading, className }: SearchBarProps) {
    const [query, setQuery] = useState('');

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (query.trim() && !isLoading) {
            onSearch(query.trim());
        }
    };

    return (
        <form
            onSubmit={handleSubmit}
            className={cn("relative w-full max-w-2xl mx-auto", className)}
        >
            <div className="relative group">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Describe the game you're looking for..."
                    disabled={isLoading}
                    className="steam-input w-full py-4 pl-12 pr-24 text-lg shadow-lg group-hover:shadow-[0_0_15px_rgba(102,192,244,0.1)] transition-shadow duration-300"
                />
                <Search
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-steam-text/50 w-5 h-5 group-focus-within:text-steam-light-blue transition-colors"
                />
                <button
                    type="submit"
                    disabled={!query.trim() || isLoading}
                    className="absolute right-2 top-1/2 -translate-y-1/2 steam-button px-4 py-1.5 text-sm disabled:opacity-0 transition-opacity"
                >
                    Search
                </button>
            </div>
        </form>
    );
}
