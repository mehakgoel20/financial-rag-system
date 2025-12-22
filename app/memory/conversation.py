class ConversationState:
    def __init__(self):
        self.last_entities = {}
        self.last_question = None

    def update(self, question: str, entities: dict):
        self.last_question = question
        self.last_entities.update(entities)

    def enrich(self, question: str) -> str:
        if "what about" in question.lower() and "company" in self.last_entities:
            return f"{question} for {self.last_entities['company']}"
        return question
