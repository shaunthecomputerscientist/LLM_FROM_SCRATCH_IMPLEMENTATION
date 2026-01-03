"""
Generate a large dataset for LLM training by creating diverse text content.
This will produce approximately 100k lines of varied text.
"""

import random

# Base story content
base_story = """I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera.

"The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication.

Poor Hermia! How little she knew of the coming metamorphosis. For Gisburn was going to Rome and Florence, but not with Hermia--that was the whole point of his retirement.

That window was the one aesthetic pleasure the poor devil had managed to retain. He liked to say that it was his one "success" in Chicago.

Gisburn was one of those unlucky artists who are too good to succeed, and not quite good enough to be great. He had in him more than average talent--far more--but not the divine spark that makes for permanent achievement.

All the women in Chicago went into ecstasies over his portraits; but Gisburn knew better than any of his critics that he had never put his best into them.

For the first time in twenty years he was going to be honest with himself, to paint what he wanted to paint instead of what would sell."""

# Template sentences for variation
templates = [
    "The artist {verb} through the {location}, contemplating his {noun}.",
    "In the quiet hours of {time}, he would {verb} about {topic}.",
    "The {adjective} landscape stretched before him, filled with {plural_noun}.",
    "She often wondered if {subject} would ever {verb} the {object}.",
    "Throughout his career, the {profession} had {verb} countless {plural_noun}.",
    "The {adjective} {noun} reminded him of {memory}.",
    "During those {adjective} years in {location}, he learned to {verb}.",
    "Every {time_period}, they would gather to discuss {topic}.",
    "The {adjective} truth was that {statement}.",
    "He couldn't help but {verb} whenever he thought about {memory}.",
]

verbs = ["wandered", "painted", "pondered", "created", "explored", "discovered", "examined", "considered", "reflected", "contemplated"]
locations = ["the studio", "Rome", "Florence", "the villa", "Chicago", "the gallery", "the countryside", "Paris", "Venice", "the museum"]
nouns = ["masterpiece", "vision", "passion", "dream", "ambition", "legacy", "craft", "technique", "inspiration", "purpose"]
adjectives = ["brilliant", "subtle", "profound", "mysterious", "elegant", "powerful", "delicate", "haunting", "vivid", "striking"]
plural_nouns = ["paintings", "memories", "portraits", "landscapes", "sketches", "masterpieces", "visions", "dreams", "moments", "scenes"]
subjects = ["the painter", "the artist", "Gisburn", "she", "Hermia", "the critic", "the patron", "the collector"]
objects = ["painting", "truth", "masterpiece", "vision", "answer", "meaning", "purpose", "beauty"]
professions = ["artist", "painter", "sculptor", "writer", "musician", "photographer", "craftsman", "creator"]
times = ["dawn", "dusk", "evening", "morning", "midnight", "twilight", "afternoon", "night"]
topics = ["art", "beauty", "truth", "life", "passion", "creativity", "meaning", "purpose", "existence", "legacy"]
time_periods = ["day", "week", "month", "year", "season", "morning", "evening", "decade"]
memories = ["his youth", "Paris", "his first exhibition", "that summer", "her smile", "the old studio", "those days", "his mentor"]
statements = [
    "art demands sacrifice",
    "success means different things to different people",
    "he had found his calling",
    "the journey mattered more than the destination",
    "creativity cannot be rushed",
    "true art comes from within",
    "genius is rarely recognized in its time",
]

print("Generating large dataset...")
print("This will create approximately 100k lines...")

output_lines = []

# 1. Add base story repeated with variations (10 times)
for i in range(10):
    output_lines.append(f"\n=== Chapter {i+1} ===\n")
    output_lines.append(base_story)
    output_lines.append("\n")

# 2. Generate varied sentences from templates
print("Generating template-based sentences...")
for _ in range(20000):
    template = random.choice(templates)
    sentence = template.format(
        verb=random.choice(verbs),
        location=random.choice(locations),
        noun=random.choice(nouns),
        adjective=random.choice(adjectives),
        plural_noun=random.choice(plural_nouns),
        subject=random.choice(subjects),
        object=random.choice(objects),
        profession=random.choice(professions),
        time=random.choice(times),
        topic=random.choice(topics),
        time_period=random.choice(time_periods),
        memory=random.choice(memories),
        statement=random.choice(statements),
    )
    output_lines.append(sentence)
    
    # Add paragraph breaks every 5-10 sentences
    if random.random() < 0.15:
        output_lines.append("\n")

# 3. Generate dialogue exchanges
print("Generating dialogue...")
dialogue_templates = [
    '"What do you think of {topic}?" she asked.',
    '"I believe that {statement}," he replied.',
    '"Tell me about {memory}," she said softly.',
    '"The {adjective} {noun} reminds me of you," he whispered.',
    '"Why did you choose to {verb}?" she wondered aloud.',
    '"In {location}, I learned that {statement}," he explained.',
    '"Every {profession} must {verb}," she insisted.',
    '"But what about {topic}?" he questioned.',
]

for _ in range(10000):
    exchange = random.sample(dialogue_templates, random.randint(2, 4))
    for line in exchange:
        formatted = line.format(
            topic=random.choice(topics),
            statement=random.choice(statements),
            memory=random.choice(memories),
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            location=random.choice(locations),
            profession=random.choice(professions),
        )
        output_lines.append(formatted)
    output_lines.append("\n")

# 4. Generate descriptive paragraphs
print("Generating descriptive paragraphs...")
for _ in range(5000):
    num_sentences = random.randint(3, 6)
    paragraph = []
    for _ in range(num_sentences):
        template = random.choice(templates)
        sentence = template.format(
            verb=random.choice(verbs),
            location=random.choice(locations),
            noun=random.choice(nouns),
            adjective=random.choice(adjectives),
            plural_noun=random.choice(plural_nouns),
            subject=random.choice(subjects),
            object=random.choice(objects),
            profession=random.choice(professions),
            time=random.choice(times),
            topic=random.choice(topics),
            time_period=random.choice(time_periods),
            memory=random.choice(memories),
            statement=random.choice(statements),
        )
        paragraph.append(sentence)
    output_lines.append(" ".join(paragraph))
    output_lines.append("\n")

# Write to file
print("Writing to file...")
with open("the-verdict.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

# Calculate statistics
total_text = "\n".join(output_lines)
print(f"\n{'='*60}")
print(f"Dataset generation complete!")
print(f"{'='*60}")
print(f"Total lines: {len(output_lines):,}")
print(f"Total characters: {len(total_text):,}")
print(f"Estimated words: {len(total_text.split()):,}")
print(f"\nFile saved to: the-verdict.txt")
print(f"{'='*60}")
