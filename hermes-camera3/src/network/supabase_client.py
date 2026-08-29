SUPABASE_BASE_URL = "https://mjrpqoinwkssmzlimwaz.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1qcnBxb2lud2tzc216bGltd2F6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODIwOTU0NzksImV4cCI6MjA5NzY3MTQ3OX0.C_1b-1dlWvgnRn9xRHhinadsfP5sHjFxQC4rcN6cruw"

class AuthState:
    def __init__(self):
        self.auth_token = ""

auth_state = AuthState()
