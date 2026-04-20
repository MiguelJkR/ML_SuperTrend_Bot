"""
ML SuperTrend v51 - Agente Financiero Inteligente (Español)
============================================================
Asistente de trading con personalidad, análisis narrativo del mercado,
recomendaciones contextuales y comunicación 100% en español.

Personalidad:
  - Profesional pero cercano
  - Analítico y basado en datos
  - Prudente con el riesgo
  - Proactivo con alertas
  - Comunica en español claro y directo

Capacidades:
  - Resumen diario narrativo (no solo números)
  - Reporte semanal con análisis de tendencias
  - Alertas de riesgo contextuales
  - Recomendaciones basadas en régimen de mercado
  - Análisis de mejores/peores horarios
  - Detección de patrones en rachas
  - Evaluación de salud del portafolio
  - Integración con GPU stats y modelo LSTM

Integración:
  - Telegram: alertas y reportes
  - Dashboard: datos para visualización
  - Trader: recomendaciones en tiempo real
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar config de timezone
try:
    from config import USER_TZ_OFFSET
except ImportError:
    USER_TZ_OFFSET = -4

_USER_TZ = timezone(timedelta(hours=USER_TZ_OFFSET))

# =====================================================================
# PERSONALIDAD DEL AGENTE
# =====================================================================
AGENT_NAME = "FinBot"
AGENT_EMOJI = "\U0001f916"

# Frases según contexto
FRASES = {
    "saludo_manana": [
        "Buenos días, jefe. Aquí tu resumen del mercado.",
        "Buen día. Revisé todo mientras dormías, esto es lo que encontré.",
        "Buenos días. El mercado ya se movió, te cuento.",
    ],
    "saludo_tarde": [
        "Buenas tardes. Aquí va el cierre parcial del día.",
        "Buenas tardes, jefe. Así vamos hasta ahora.",
    ],
    "saludo_noche": [
        "Buenas noches. Cierre del día listo.",
        "Fin del día. Aquí el resumen completo.",
    ],
    "ganancia_fuerte": [
        "Excelente día. Los números hablan por sí solos.",
        "Gran jornada. El modelo está afinado.",
        "Día productivo. Así se hace.",
    ],
    "ganancia_leve": [
        "Día positivo, aunque moderado. Cada pip suma.",
        "Verde ligero. La consistencia es la clave.",
    ],
    "perdida_leve": [
        "Día ligeramente rojo. Nada fuera de lo normal.",
        "Pequeña pérdida. El mercado no siempre coopera.",
    ],
    "perdida_fuerte": [
        "Día difícil. Revisa las condiciones antes de mañana.",
        "Jornada complicada. Sugiero reducir tamaño mañana.",
    ],
    "sin_trades": [
        "Sin operaciones hoy. A veces la mejor trade es no operar.",
        "Día sin actividad. El mercado no dio condiciones.",
    ],
    "racha_ganadora": [
        "Racha ganadora activa. No te confíes, mantén disciplina.",
        "Vamos en racha. Recuerda: el mercado siempre puede girar.",
    ],
    "racha_perdedora": [
        "Racha perdedora. Considera reducir tamaño al 50%.",
        "Varias pérdidas seguidas. Pausa y revisa la estrategia.",
    ],
    "drawdown_alerta": [
        "ALERTA: Drawdown significativo. Protege el capital.",
        "Drawdown elevado. Prioridad #1: preservar capital.",
    ],
    "mercado_volatil": [
        "El mercado está muy volátil. Reduce exposición.",
        "Alta volatilidad detectada. Cuidado con los stops.",
    ],
    "mercado_lateral": [
        "Mercado lateral. Las señales de tendencia son menos fiables.",
        "Rango estrecho. Mejor esperar una ruptura clara.",
    ],
    "mercado_tendencia": [
        "Tendencia clara detectada. Buen momento para seguirla.",
        "El mercado tiene dirección. Aprovecha con disciplina.",
    ],
}


def _frase(categoria: str) -> str:
    """Seleccionar frase según hora actual para variedad."""
    import random
    frases = FRASES.get(categoria, [""])
    # Usar hora como seed para consistencia dentro de la misma hora
    idx = datetime.now(_USER_TZ).hour % len(frases)
    return frases[idx]


class FinancialAdvisor:
    """
    Agente financiero inteligente con personalidad en español.
    Analiza, aconseja y reporta en lenguaje natural.
    """

    def __init__(self, telegram_token: str = "", telegram_chat_id: str = "",
                 data_dir: str = None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "advisor_data"
        )
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # Tracking
        self.trades_history: List[Dict] = []
        self.daily_pnl_log: Dict = {}
        self.win_loss_streak = {"wins": 0, "losses": 0}
        self.peak_balance = 0.0
        self.regime_history: Dict[str, str] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        self.session_stats: Dict[str, Dict] = {}  # Stats por sesión (ASIAN, LONDON, etc.)
        self.instrument_stats: Dict[str, Dict] = {}  # Stats por par
        self.daily_notes: List[str] = []  # Notas del día

        self.load()

    # ─────────────────────────────────────────────────────────
    # TELEGRAM
    # ─────────────────────────────────────────────────────────
    def send_telegram_sync(self, message: str, parse_mode: str = "HTML") -> bool:
        """Enviar mensaje por Telegram (síncrono, sin dependencias async)."""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.debug("Telegram no configurado, mensaje omitido")
            return False
        try:
            import urllib.request
            import urllib.parse
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": parse_mode
            }).encode('utf-8')
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("ok", False)
        except Exception as e:
            logger.warning(f"Telegram error: {e}")
            return False

    # Alias async para compatibilidad
    async def send_telegram(self, message: str, parse_mode: str = "HTML") -> bool:
        return self.send_telegram_sync(message, parse_mode)

    # ─────────────────────────────────────────────────────────
    # REGISTRO DE TRADES
    # ─────────────────────────────────────────────────────────
    def record_trade(self, trade_data: Dict):
        """
        Registrar un trade completado y generar alerta contextual.

        trade_data: {
            instrument, entry_time, exit_time, entry_price, exit_price,
            pnl, pnl_pct, volume, direction, regime, balance_after,
            session, attention_confidence
        }
        """
        self.trades_history.append(trade_data)

        pnl = trade_data.get("pnl", 0)
        instrument = trade_data.get("instrument", "???")
        direction = trade_data.get("direction", "???")
        session = trade_data.get("session", "")

        # Actualizar racha
        if pnl > 0:
            self.win_loss_streak["wins"] += 1
            self.win_loss_streak["losses"] = 0
        else:
            self.win_loss_streak["losses"] += 1
            self.win_loss_streak["wins"] = 0

        # Actualizar balance pico
        balance = trade_data.get("balance_after", 0)
        if balance > self.peak_balance:
            self.peak_balance = balance

        # Actualizar stats por sesión
        if session:
            if session not in self.session_stats:
                self.session_stats[session] = {"pnl": 0, "trades": 0, "wins": 0}
            self.session_stats[session]["pnl"] += pnl
            self.session_stats[session]["trades"] += 1
            if pnl > 0:
                self.session_stats[session]["wins"] += 1

        # Actualizar stats por instrumento
        if instrument not in self.instrument_stats:
            self.instrument_stats[instrument] = {"pnl": 0, "trades": 0, "wins": 0}
        self.instrument_stats[instrument]["pnl"] += pnl
        self.instrument_stats[instrument]["trades"] += 1
        if pnl > 0:
            self.instrument_stats[instrument]["wins"] += 1

        # Generar alerta en español
        emoji = "\u2705" if pnl > 0 else "\u274c"
        dir_es = "COMPRA" if direction == "LONG" else "VENTA"

        msg_lines = [
            f"{emoji} <b>Trade Cerrado</b>",
            f"Par: {instrument} | {dir_es}",
            f"PnL: <b>{pnl:+.2f}</b> ({trade_data.get('pnl_pct', 0):+.2f}%)",
            f"Balance: {balance:,.2f}",
        ]

        # Contexto de racha
        if self.win_loss_streak["wins"] >= 3:
            msg_lines.append(f"\U0001f525 Racha ganadora: {self.win_loss_streak['wins']} seguidas")
        elif self.win_loss_streak["losses"] >= 3:
            msg_lines.append(f"\u26a0\ufe0f Racha perdedora: {self.win_loss_streak['losses']} seguidas")
            msg_lines.append(f"<i>{_frase('racha_perdedora')}</i>")

        # Contexto de atención del modelo
        attn_conf = trade_data.get("attention_confidence")
        if attn_conf is not None:
            msg_lines.append(f"Confianza modelo: {attn_conf:.0%}")

        self.send_telegram_sync("\n".join(msg_lines))

    # ─────────────────────────────────────────────────────────
    # RESUMEN DIARIO (narrativo)
    # ─────────────────────────────────────────────────────────
    def get_daily_summary(self) -> str:
        """Generar resumen diario narrativo en español."""
        ahora = datetime.now(_USER_TZ)
        hoy = ahora.strftime("%Y-%m-%d")
        dia_nombre = ["Lunes", "Martes", "Miércoles", "Jueves",
                      "Viernes", "Sábado", "Domingo"][ahora.weekday()]
        hora = ahora.hour

        # Saludo según hora
        if hora < 12:
            saludo = _frase("saludo_manana")
        elif hora < 18:
            saludo = _frase("saludo_tarde")
        else:
            saludo = _frase("saludo_noche")

        # Trades del día
        today_trades = []
        for t in self.trades_history:
            try:
                exit_t = t.get("exit_time", "")
                if isinstance(exit_t, str) and exit_t.startswith(hoy):
                    today_trades.append(t)
                elif hasattr(exit_t, 'strftime') and exit_t.strftime("%Y-%m-%d") == hoy:
                    today_trades.append(t)
            except Exception:
                pass

        daily_pnl = sum(t.get("pnl", 0) for t in today_trades)
        num_trades = len(today_trades)
        wins = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
        losses = num_trades - wins
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

        # Contexto narrativo
        if num_trades == 0:
            contexto = _frase("sin_trades")
        elif daily_pnl > 50:
            contexto = _frase("ganancia_fuerte")
        elif daily_pnl > 0:
            contexto = _frase("ganancia_leve")
        elif daily_pnl > -30:
            contexto = _frase("perdida_leve")
        else:
            contexto = _frase("perdida_fuerte")

        # Balance actual
        balance = today_trades[-1].get("balance_after", 0) if today_trades else self.peak_balance

        # Construir resumen
        lines = [
            f"{AGENT_EMOJI} <b>{AGENT_NAME} - Resumen {dia_nombre} {hoy}</b>",
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
            f"<i>{saludo}</i>",
            "",
            f"\U0001f4ca <b>Resultado del día:</b>",
        ]

        if num_trades > 0:
            pnl_emoji = "\U0001f7e2" if daily_pnl >= 0 else "\U0001f534"
            lines.extend([
                f"{pnl_emoji} PnL: <b>{daily_pnl:+.2f}</b>",
                f"Operaciones: {num_trades} (\u2705{wins} / \u274c{losses})",
                f"Tasa de acierto: {win_rate:.0f}%",
                f"Balance: {balance:,.2f}",
            ])

            # Mejor y peor trade
            if today_trades:
                best = max(today_trades, key=lambda t: t.get("pnl", 0))
                worst = min(today_trades, key=lambda t: t.get("pnl", 0))
                lines.append(f"Mejor: {best.get('instrument', '?')} ({best.get('pnl', 0):+.2f})")
                if num_trades > 1:
                    lines.append(f"Peor: {worst.get('instrument', '?')} ({worst.get('pnl', 0):+.2f})")
        else:
            lines.append(f"Sin operaciones hoy.")

        lines.append("")
        lines.append(f"<i>{contexto}</i>")

        # Racha
        if self.win_loss_streak["wins"] >= 2:
            lines.append(f"\n\U0001f525 Racha ganadora: {self.win_loss_streak['wins']} seguidas")
        elif self.win_loss_streak["losses"] >= 2:
            lines.append(f"\n\u26a0\ufe0f Racha perdedora: {self.win_loss_streak['losses']} seguidas")

        # Drawdown
        if self.peak_balance > 0 and balance > 0:
            dd = (1 - balance / self.peak_balance) * 100
            if dd > 3:
                lines.append(f"\n\U0001f4c9 Drawdown actual: {dd:.1f}%")

        # Régimen de mercado
        if self.regime_history:
            lines.append("\n<b>Régimen actual:</b>")
            for instr, reg in list(self.regime_history.items())[:3]:
                reg_emoji = {"TRENDING": "\U0001f4c8", "RANGING": "\u2194\ufe0f", "VOLATILE": "\u26a1"}.get(reg, "\u2753")
                reg_es = {"TRENDING": "Tendencia", "RANGING": "Lateral",
                          "VOLATILE": "Volátil", "MEAN_REVERT": "Reversión"}.get(reg, reg)
                lines.append(f"  {reg_emoji} {instr}: {reg_es}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # REPORTE SEMANAL
    # ─────────────────────────────────────────────────────────
    def get_weekly_report(self) -> str:
        """Generar reporte semanal detallado en español."""
        ahora = datetime.now(_USER_TZ)
        inicio_semana = ahora - timedelta(days=7)

        week_trades = []
        for t in self.trades_history:
            try:
                exit_t = t.get("exit_time", "")
                if isinstance(exit_t, str):
                    dt = datetime.fromisoformat(exit_t)
                else:
                    dt = exit_t
                if dt.date() >= inicio_semana.date():
                    week_trades.append(t)
            except Exception:
                pass

        if not week_trades:
            return (f"{AGENT_EMOJI} <b>{AGENT_NAME} - Reporte Semanal</b>\n"
                    f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                    f"Sin operaciones esta semana.")

        weekly_pnl = sum(t.get("pnl", 0) for t in week_trades)
        num_trades = len(week_trades)
        wins = sum(1 for t in week_trades if t.get("pnl", 0) > 0)
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.get("pnl", 0) for t in week_trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in week_trades if t.get("pnl", 0) < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # PnL por día
        daily_pnls = {}
        for t in week_trades:
            try:
                exit_t = t.get("exit_time", "")
                if isinstance(exit_t, str):
                    day = exit_t[:10]
                else:
                    day = exit_t.strftime("%Y-%m-%d")
                daily_pnls[day] = daily_pnls.get(day, 0) + t.get("pnl", 0)
            except Exception:
                pass

        # Mejor sesión
        session_pnl = {}
        for t in week_trades:
            s = t.get("session", "OTRO")
            session_pnl[s] = session_pnl.get(s, 0) + t.get("pnl", 0)
        best_session = max(session_pnl.items(), key=lambda x: x[1]) if session_pnl else ("N/A", 0)

        # Mejor par
        pair_pnl = {}
        for t in week_trades:
            p = t.get("instrument", "?")
            pair_pnl[p] = pair_pnl.get(p, 0) + t.get("pnl", 0)
        best_pair = max(pair_pnl.items(), key=lambda x: x[1]) if pair_pnl else ("N/A", 0)

        # Mejor hora
        best_hour = self._get_best_trading_hours()

        pnl_emoji = "\U0001f7e2" if weekly_pnl >= 0 else "\U0001f534"

        lines = [
            f"{AGENT_EMOJI} <b>{AGENT_NAME} - Reporte Semanal</b>",
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
            f"Periodo: {inicio_semana.strftime('%d/%m')} - {ahora.strftime('%d/%m/%Y')}",
            "",
            f"\U0001f4ca <b>Rendimiento:</b>",
            f"{pnl_emoji} PnL total: <b>{weekly_pnl:+.2f}</b>",
            f"Operaciones: {num_trades} (\u2705{wins} / \u274c{num_trades - wins})",
            f"Tasa de acierto: {win_rate:.0f}%",
            f"Factor de ganancia: {profit_factor:.2f}",
            "",
            f"\U0001f3c6 <b>Mejor sesión:</b> {best_session[0]} ({best_session[1]:+.2f})",
            f"\U0001f4b0 <b>Mejor par:</b> {best_pair[0]} ({best_pair[1]:+.2f})",
            f"\u23f0 <b>Mejor hora:</b> {best_hour}",
        ]

        # PnL diario visual
        if daily_pnls:
            lines.append(f"\n\U0001f4c5 <b>PnL por día:</b>")
            dias_es = {"Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié",
                       "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb", "Sunday": "Dom"}
            for day, pnl in sorted(daily_pnls.items()):
                try:
                    dt = datetime.strptime(day, "%Y-%m-%d")
                    dia_en = dt.strftime("%A")
                    dia = dias_es.get(dia_en, dia_en[:3])
                except Exception:
                    dia = day[-5:]
                bar = "\U0001f7e9" * max(1, min(int(abs(pnl) / 10), 8)) if pnl >= 0 else "\U0001f7e5" * max(1, min(int(abs(pnl) / 10), 8))
                lines.append(f"  {dia}: {bar} {pnl:+.2f}")

        # Recomendación para la próxima semana
        lines.append(f"\n\U0001f4a1 <b>Recomendación:</b>")
        rec = self.get_recommendation()
        lines.append(f"<i>{rec['reason']}</i>")
        if rec["adjustments"]:
            for adj in rec["adjustments"][:3]:
                lines.append(f"  \u2022 {adj}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # ALERTAS DE RIESGO
    # ─────────────────────────────────────────────────────────
    def check_risk_alerts(self, current_balance: float, peak_balance: float = None,
                          open_trades: List[Dict] = None) -> List[Dict]:
        """Verificar condiciones de riesgo y generar alertas en español."""
        if peak_balance is None:
            peak_balance = self.peak_balance
        if open_trades is None:
            open_trades = []

        alerts = []

        # Drawdown
        if peak_balance > 0:
            dd_pct = (1 - current_balance / peak_balance) * 100
            if dd_pct > 10:
                alerts.append({
                    "type": "DRAWDOWN_CRITICO",
                    "severity": "HIGH",
                    "message": f"\U0001f6a8 CRÍTICO: Drawdown al {dd_pct:.1f}%. Pausa el trading AHORA.",
                })
            elif dd_pct > 8:
                alerts.append({
                    "type": "DRAWDOWN_ALTO",
                    "severity": "HIGH",
                    "message": f"\u26a0\ufe0f Drawdown al {dd_pct:.1f}%. Reduce posiciones al mínimo.",
                })
            elif dd_pct > 5:
                alerts.append({
                    "type": "DRAWDOWN_MEDIO",
                    "severity": "MEDIUM",
                    "message": f"\U0001f4c9 Drawdown al {dd_pct:.1f}%. Monitorea de cerca.",
                })

        # Racha perdedora
        if self.win_loss_streak["losses"] >= 5:
            alerts.append({
                "type": "RACHA_CRITICA",
                "severity": "HIGH",
                "message": f"\U0001f6d1 {self.win_loss_streak['losses']} pérdidas consecutivas. PAUSA obligatoria.",
            })
        elif self.win_loss_streak["losses"] >= 3:
            alerts.append({
                "type": "RACHA_PERDEDORA",
                "severity": "MEDIUM",
                "message": f"\U0001f4c9 {self.win_loss_streak['losses']} pérdidas seguidas. Reduce tamaño al 50%.",
            })

        # Exposición total
        total_risk = sum(t.get("risk_amount", 0) for t in open_trades)
        if current_balance > 0 and total_risk > 0:
            risk_pct = total_risk / current_balance * 100
            if risk_pct > 10:
                alerts.append({
                    "type": "SOBREEXPOSICION",
                    "severity": "HIGH",
                    "message": f"\U0001f534 Exposición total: {risk_pct:.1f}% del balance. Muy alto.",
                })
            elif risk_pct > 6:
                alerts.append({
                    "type": "EXPOSICION_ALTA",
                    "severity": "MEDIUM",
                    "message": f"\u26a0\ufe0f Exposición: {risk_pct:.1f}%. Considera cerrar alguna posición.",
                })

        # Muchas posiciones abiertas
        if len(open_trades) > 4:
            alerts.append({
                "type": "MUCHAS_POSICIONES",
                "severity": "MEDIUM",
                "message": f"\U0001f4ca {len(open_trades)} posiciones abiertas. Difícil gestionar tantas.",
            })

        return alerts

    # ─────────────────────────────────────────────────────────
    # RECOMENDACIONES
    # ─────────────────────────────────────────────────────────
    def get_recommendation(self) -> Dict:
        """Recomendación actual basada en todos los factores."""
        adjustments = []
        recommendation = "NORMAL"
        reason = "Condiciones normales. Opera con disciplina."

        # Racha
        if self.win_loss_streak["losses"] >= 5:
            recommendation = "PAUSA"
            reason = "Demasiadas pérdidas consecutivas. Detén el trading y revisa la estrategia."
            adjustments.append(f"{self.win_loss_streak['losses']} pérdidas seguidas \u2014 pausa obligatoria")
        elif self.win_loss_streak["losses"] >= 3:
            recommendation = "REDUCIR_RIESGO"
            reason = "Racha perdedora activa. Reduce el tamaño de posición al 50%."
            adjustments.append(f"{self.win_loss_streak['losses']} pérdidas seguidas \u2014 reducir tamaño")
        elif self.win_loss_streak["wins"] >= 5:
            adjustments.append(f"Racha de {self.win_loss_streak['wins']} victorias \u2014 no te confíes")

        # Régimen de mercado
        regimes = list(self.regime_history.values())
        if regimes:
            volatile_count = sum(1 for r in regimes if r in ("VOLATILE", "CHOPPY"))
            if volatile_count > len(regimes) * 0.5:
                if recommendation == "NORMAL":
                    recommendation = "REDUCIR_RIESGO"
                    reason = "Mercado predominantemente volátil. Reduce exposición."
                adjustments.append("Múltiples pares en régimen volátil")

        # Variedad de regímenes (cambios frecuentes = mercado inestable)
        if len(self.trades_history) >= 10:
            recent = self.trades_history[-10:]
            unique_regimes = len(set(t.get("regime", "") for t in recent))
            if unique_regimes > 3:
                adjustments.append("Alta rotación de regímenes \u2014 mercado inestable")

        # Hora del día
        ahora = datetime.now(_USER_TZ)
        if ahora.weekday() >= 5:
            recommendation = "PAUSA"
            reason = "Fin de semana. Los mercados principales están cerrados."
            adjustments.append("Fin de semana \u2014 sin operar")
        elif ahora.weekday() == 4 and ahora.hour >= 16:
            adjustments.append("Viernes tarde \u2014 reduce exposición antes del cierre")

        return {
            "recommendation": recommendation,
            "recommendation_es": {
                "NORMAL": "Operar normal",
                "REDUCIR_RIESGO": "Reducir riesgo",
                "PAUSA": "Pausar trading",
            }.get(recommendation, recommendation),
            "reason": reason,
            "adjustments": adjustments,
        }

    # ─────────────────────────────────────────────────────────
    # ANÁLISIS DE MERCADO NARRATIVO
    # ─────────────────────────────────────────────────────────
    def analizar_mercado(self, data: Dict) -> str:
        """
        Generar análisis narrativo del mercado actual.

        data: {
            instruments: [{name, price, change_pct, regime, rsi, adx, lstm_prob}],
            session: str,
            open_exchanges: [str],
            active_zones: [str],
            model_accuracy: float,
            gpu_status: dict,
        }
        """
        instruments = data.get("instruments", [])
        session = data.get("session", "???")
        zones = data.get("active_zones", [])
        exchanges = data.get("open_exchanges", [])
        accuracy = data.get("model_accuracy", 0)

        session_es = {
            "ASIAN": "Asiática", "LONDON": "Londres", "NEW_YORK": "Nueva York",
            "LONDON_ASIAN": "Londres-Asia", "LONDON_NY": "Londres-NY",
            "OFF_HOURS": "Fuera de horario", "WEEKEND": "Fin de semana"
        }.get(session, session)

        lines = [
            f"{AGENT_EMOJI} <b>{AGENT_NAME} - Análisis de Mercado</b>",
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
            f"Sesión: <b>{session_es}</b>",
        ]

        if zones:
            lines.append(f"Zonas activas: {' | '.join(zones)}")
        if exchanges:
            lines.append(f"Bolsas abiertas: {', '.join(exchanges[:5])}")

        lines.append("")

        # Análisis por instrumento
        for inst in instruments[:5]:
            name = inst.get("name", "?")
            regime = inst.get("regime", "?")
            rsi = inst.get("rsi", 50)
            adx = inst.get("adx", 20)
            lstm_prob = inst.get("lstm_prob", 0.5)
            change = inst.get("change_pct", 0)

            # Determinar sesgo
            if lstm_prob > 0.65:
                sesgo = "\U0001f4c8 Alcista"
            elif lstm_prob < 0.35:
                sesgo = "\U0001f4c9 Bajista"
            else:
                sesgo = "\u2194\ufe0f Neutral"

            # RSI contexto
            if rsi > 70:
                rsi_ctx = "sobrecomprado"
            elif rsi < 30:
                rsi_ctx = "sobrevendido"
            else:
                rsi_ctx = "neutral"

            # Fuerza de tendencia
            if adx > 40:
                fuerza = "fuerte"
            elif adx > 25:
                fuerza = "moderada"
            else:
                fuerza = "débil"

            regime_es = {"TRENDING": "Tendencia", "RANGING": "Lateral",
                         "VOLATILE": "Volátil", "MEAN_REVERT": "Reversión"}.get(regime, regime)

            lines.append(f"<b>{name}</b>: {sesgo}")
            lines.append(f"  Régimen: {regime_es} | RSI: {rsi:.0f} ({rsi_ctx}) | ADX: {adx:.0f} ({fuerza})")
            lines.append(f"  Modelo: {lstm_prob:.0%} UP | Cambio: {change:+.2f}%")
            lines.append("")

        # Estado del modelo
        if accuracy > 0:
            acc_emoji = "\U0001f3af" if accuracy > 0.55 else "\u26a0\ufe0f"
            lines.append(f"{acc_emoji} Precisión modelo: {accuracy:.1%}")

        # GPU stats
        gpu = data.get("gpu_status", {})
        if gpu.get("cuda_available"):
            lines.append(f"\U0001f5a5\ufe0f GPU: {gpu.get('temp_c', '?')}\u00b0C | VRAM: {gpu.get('vram_pct', '?')}%")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────
    def _get_best_trading_hours(self) -> str:
        """Analizar mejores horas de trading."""
        if not self.trades_history:
            return "N/A"

        hour_pnl = {}
        for trade in self.trades_history:
            try:
                exit_t = trade.get("exit_time", "")
                if isinstance(exit_t, str):
                    dt = datetime.fromisoformat(exit_t)
                else:
                    dt = exit_t
                hour = dt.hour
                hour_pnl[hour] = hour_pnl.get(hour, 0) + trade.get("pnl", 0)
            except Exception:
                pass

        if not hour_pnl:
            return "N/A"

        best = max(hour_pnl.items(), key=lambda x: x[1])
        return f"{best[0]:02d}:00 UTC ({best[1]:+.2f})"

    def get_best_trading_hours(self) -> Dict:
        """Estadísticas detalladas por hora."""
        if not self.trades_history:
            return {"best_hour": None, "worst_hour": None, "data": {}}

        hour_stats = {}
        for trade in self.trades_history:
            try:
                exit_t = trade.get("exit_time", "")
                if isinstance(exit_t, str):
                    dt = datetime.fromisoformat(exit_t)
                else:
                    dt = exit_t
                hour = dt.hour
                if hour not in hour_stats:
                    hour_stats[hour] = {"pnl": 0, "count": 0}
                hour_stats[hour]["pnl"] += trade.get("pnl", 0)
                hour_stats[hour]["count"] += 1
            except Exception:
                pass

        if not hour_stats:
            return {"best_hour": None, "worst_hour": None, "data": {}}

        best = max(hour_stats.items(), key=lambda x: x[1]["pnl"])
        worst = min(hour_stats.items(), key=lambda x: x[1]["pnl"])
        return {
            "best_hour": f"{best[0]:02d}:00 UTC",
            "worst_hour": f"{worst[0]:02d}:00 UTC",
            "data": {str(h): v for h, v in hour_stats.items()}
        }

    def update_regime(self, instrument: str, regime: str):
        """Actualizar régimen de mercado para un instrumento."""
        self.regime_history[instrument] = regime

    def get_session_performance(self) -> Dict:
        """Rendimiento por sesión de trading."""
        result = {}
        for session, stats in self.session_stats.items():
            trades = stats["trades"]
            result[session] = {
                "pnl": round(stats["pnl"], 2),
                "trades": trades,
                "wins": stats["wins"],
                "win_rate": round(stats["wins"] / trades * 100, 1) if trades > 0 else 0,
            }
        return result

    def get_instrument_performance(self) -> Dict:
        """Rendimiento por par de divisas."""
        result = {}
        for instr, stats in self.instrument_stats.items():
            trades = stats["trades"]
            result[instr] = {
                "pnl": round(stats["pnl"], 2),
                "trades": trades,
                "wins": stats["wins"],
                "win_rate": round(stats["wins"] / trades * 100, 1) if trades > 0 else 0,
            }
        return result

    def get_dashboard_data(self) -> Dict:
        """Datos completos para el dashboard."""
        ahora = datetime.now(_USER_TZ)
        rec = self.get_recommendation()
        return {
            "agent_name": AGENT_NAME,
            "timestamp": ahora.isoformat(),
            "recommendation": rec,
            "streak": self.win_loss_streak.copy(),
            "peak_balance": self.peak_balance,
            "total_trades": len(self.trades_history),
            "session_performance": self.get_session_performance(),
            "instrument_performance": self.get_instrument_performance(),
            "regime_history": self.regime_history.copy(),
            "best_hours": self.get_best_trading_hours(),
        }

    # ─────────────────────────────────────────────────────────
    # PERSISTENCIA
    # ─────────────────────────────────────────────────────────
    def save(self):
        """Guardar estado del agente a disco."""
        try:
            trades_file = os.path.join(self.data_dir, "trades_history.json")
            with open(trades_file, 'w', encoding='utf-8') as f:
                json.dump(self.trades_history, f, indent=2, default=str, ensure_ascii=False)

            state_file = os.path.join(self.data_dir, "advisor_state.json")
            state = {
                "win_loss_streak": self.win_loss_streak,
                "peak_balance": self.peak_balance,
                "regime_history": self.regime_history,
                "session_stats": self.session_stats,
                "instrument_stats": self.instrument_stats,
            }
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            logger.info(f"Estado del agente guardado en {self.data_dir}")
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")

    def load(self):
        """Cargar estado del agente desde disco."""
        try:
            trades_file = os.path.join(self.data_dir, "trades_history.json")
            if os.path.exists(trades_file):
                with open(trades_file, 'r', encoding='utf-8') as f:
                    self.trades_history = json.load(f)

            state_file = os.path.join(self.data_dir, "advisor_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.win_loss_streak = state.get("win_loss_streak", {"wins": 0, "losses": 0})
                    self.peak_balance = state.get("peak_balance", 0)
                    self.regime_history = state.get("regime_history", {})
                    self.session_stats = state.get("session_stats", {})
                    self.instrument_stats = state.get("instrument_stats", {})

            logger.info(f"Estado del agente cargado desde {self.data_dir}")
        except Exception as e:
            logger.warning(f"Error cargando estado: {e}")


if __name__ == "__main__":
    advisor = FinancialAdvisor()
    print(advisor.get_daily_summary())
    print()
    print(advisor.get_recommendation())
