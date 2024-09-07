class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:
        self.phoneme_table = phoneme_table
        self.vocabulary = vocabulary
        self.best_state = None
        self.visited_states = set()  # To track already visited states

    def asr_corrector(self, environment):
        # Initialize with the current state
        current_state = environment.init_state
        self.best_state = current_state
        best_cost = environment.compute_cost(current_state)

        # Add the initial state to the visited set
        self.visited_states.add(current_state)

        # Set stopping criteria
        max_iterations = 100
        iterations = 0

        while iterations < max_iterations:
            # First pass: Generate phoneme-corrected neighbors
            phoneme_corrected_neighbors = self.generate_phoneme_neighbors(current_state)

            has_update = False

            # Evaluate phoneme-corrected neighbors
            for neighbor in phoneme_corrected_neighbors:
                if neighbor in self.visited_states:
                    continue  # Skip if the state has already been visited

                cost = environment.compute_cost(neighbor)
                if cost < best_cost:
                    self.best_state = neighbor
                    best_cost = cost
                    current_state = neighbor  # Move to this better state
                    has_update = True

                # Mark the neighbor as visited
                self.visited_states.add(neighbor)

            # If phoneme corrections were made, generate word-insertion neighbors based on the best phoneme-corrected state
            if has_update:
                word_inserted_neighbors = self.generate_word_insertion_neighbors(current_state)

                # Evaluate word-inserted neighbors
                for neighbor in word_inserted_neighbors:
                    if neighbor in self.visited_states:
                        continue  # Skip if the state has already been visited

                    cost = environment.compute_cost(neighbor)
                    if cost < best_cost:
                        self.best_state = neighbor
                        best_cost = cost
                        current_state = neighbor  # Move to this better state

                    # Mark the neighbor as visited
                    self.visited_states.add(neighbor)

            iterations += 1

    def generate_phoneme_neighbors(self, state):
        neighbors = []
        words_in_state = state.split()

        # Apply phoneme corrections
        # Allow substitutions anywhere in the word based on the phoneme table
        for i, word in enumerate(words_in_state):

            # Iterate through the phoneme table to find replacements
            for correct_phoneme, incorrect_phonemes in self.phoneme_table.items():
                for incorrect_phoneme in incorrect_phonemes:
                    # Look for the incorrect phoneme anywhere in the word
                    start = 0
                    while start < len(word):
                        index = word.find(incorrect_phoneme, start)
                        
                        if index != -1:
                            new_word = word[:index] + correct_phoneme + word[index + len(incorrect_phoneme):]
                            new_state = ' '.join(words_in_state[:i] + [new_word] + words_in_state[i+1:])
                            neighbors.append(new_state)
                            # Debugging print to show the newly generated neighbor
                            #print(f"Generated neighbor: {new_state}")
                            
                            start = index + len(incorrect_phoneme)
                        else:
                            break

        return neighbors

    def generate_word_insertion_neighbors(self, state):
        neighbors = []
        words_in_state = state.split()

        # Insert words at the beginning if the sentence does not start with an article or preposition
        #if words_in_state[0] not in ["THE", "A", "AN", "TO", "IN", "ON", "HE", "SHE", "I", "IT", "THEY", "IF", "WHAT", "WHICH", "WHERE", "THERE", "THIS"]:
        for word in self.vocabulary:
            neighbors.append(word + " " + state)  # Add to the beginning

        # Insert words at the end if the sentence ends with a conjunction or preposition
        #if words_in_state[-1] in ["AND", "OR", "BUT", "TO", "WITH", "IN", "FOR", "THE", "A", "AN", "HAS", "HAD", "HAVE"]:
        for word in self.vocabulary:
            neighbors.append(state + " " + word)  # Add to the end

        # Print out all neighbors generated
        #print(f"Generated neighbors for state '{state}':")
        #for neighbor in neighbors:
        #    print(f" - {neighbor}")
            
        return neighbors

