import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  size = 'md',
  icon,
}) => {
  const variantStyles = {
    default: 'bg-slate-700/60 text-slate-200 border-slate-600',
    success: 'bg-emerald-950/70 text-emerald-300 border-emerald-700/60',
    warning: 'bg-amber-950/70 text-amber-300 border-amber-700/60',
    danger: 'bg-rose-950/70 text-rose-300 border-rose-700/60',
    info: 'bg-sky-950/70 text-sky-300 border-sky-700/60',
    purple: 'bg-purple-950/70 text-purple-300 border-purple-700/60',
  };

  const sizeStyles = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-xs px-2.5 py-1 gap-1.5',
    lg: 'text-sm px-3 py-1.5 gap-2',
  };

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border shadow-sm ${variantStyles[variant]} ${sizeStyles[size]}`}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      <span>{children}</span>
    </span>
  );
};
