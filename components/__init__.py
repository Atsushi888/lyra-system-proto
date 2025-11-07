# from components.preflight import PreflightChecker
# from components.debug_panel import DebugPanel
# from components.chat_log import ChatLog
# from components.player_input import PlayerInput

# __all__ = ["PreflightChecker", "DebugPanel", "ChatLog", "PlayerInput" ]
self.preflight = preflight.PreflightChecker(self.openai_key, self.openrouter_key)
self.debug_panel = debug_panel.DebugPanel()
self.chat_log = chat_log.ChatLog(self.partner_name, self.DISPLAY_LIMIT)
self.player_input = player_input.PlayerInput()
