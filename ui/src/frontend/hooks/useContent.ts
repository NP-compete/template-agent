import { useMemo } from 'react';
import { useSelector } from 'react-redux';
import type { RootState } from '../redux/store';
import type { ContentConfig } from '../services/config.service';

/** Mirrors the server's `content` defaults so copy renders before /api/config
 * resolves, and still renders if it fails. */
const FALLBACK: ContentConfig = {
  greeting: 'Hey {user}! 👋',
  greeting_anonymous: 'Hey there! 👋',
  subtitle: '{agent} is ready to help. What would you like to explore today?',
  quick_prompts_heading: "Let's try some quick prompts 🚀",
  quick_prompts: [
    'What can {agent} do for me?',
    'Help me analyze a dataset',
    'Write a query to find anomalies',
    'Summarize the key findings from this data',
  ],
  input_placeholder: 'Ask {agent} anything...',
  welcome_heading: 'Welcome to {agent}!',
  welcome_subtitle: 'How can I help you today?',
};

function substitute(template: string, tokens: Record<string, string>): string {
  return template.replace(/\{(agent|user)\}/g, (_, key: string) => tokens[key] ?? '');
}

export interface ResolvedContent extends Omit<ContentConfig, 'quick_prompts'> {
  quick_prompts: string[];
  /** The greeting already picked for the current user. */
  resolved_greeting: string;
}

/** Landing copy with `{agent}` / `{user}` resolved.
 *
 * Copy lives in `config/ui/settings.yaml` under `content:`; this is the only
 * place components should read it from. */
export function useContent(): ResolvedContent {
  const content = useSelector((state: RootState) => state.config.content);

  return useMemo(() => {
    const cfg = content ?? FALLBACK;
    const agent = window.APP_DATA?.agentName || 'Agent';
    const user = window.USER_DATA?.displayName || window.USER_DATA?.given_name || '';
    const tokens = { agent, user };
    const render = (s: string) => substitute(s ?? '', tokens);

    return {
      greeting: render(cfg.greeting),
      greeting_anonymous: render(cfg.greeting_anonymous),
      subtitle: render(cfg.subtitle),
      quick_prompts_heading: render(cfg.quick_prompts_heading),
      quick_prompts: (cfg.quick_prompts ?? []).map(render),
      input_placeholder: render(cfg.input_placeholder),
      welcome_heading: render(cfg.welcome_heading),
      welcome_subtitle: render(cfg.welcome_subtitle),
      resolved_greeting: user ? render(cfg.greeting) : render(cfg.greeting_anonymous),
    };
  }, [content]);
}
