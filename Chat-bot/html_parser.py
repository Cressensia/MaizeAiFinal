from bs4 import BeautifulSoup

def extract_data_from_table(table):
    # Concatenate the content of the table
    table_data = []
    for row in table.find_all('tr'):
        row_data = [cell.get_text(strip=True) for cell in row.find_all('td')]
        table_data.append(' '.join(row_data))
    return ' '.join(table_data)

html_content = """
...  # (your HTML content here)
"""

soup = BeautifulSoup(html_content, 'html.parser')

# Extract data inside the comment<!-- PRINTING START HERE --> <!-- PRINTING ENDS HERE -->
printing_section = soup.find(string=lambda text: 'PRINTING STARTS HERE' in str(text))
if printing_section:
    # Extract data inside the comment
    printing_data = printing_section.find_next(string=lambda text: 'PRINTING ENDS HERE' in str(text)).find_all_next()

    # Initialize variables to store questions and answers
    questions = []
    answers = []

    current_question = None
    current_answers = []

    for element in printing_data:
        # Check if the element is an <h> tag
        if element.name and element.name.startswith('h'):
            # If we have a current question, save it along with the accumulated answers
            if current_question:
                questions.append(current_question)
                answers.append(' '.join(current_answers))
                current_answers = []

            # Set the current <h> tag as the question
            current_question = element.get_text()

        # Check if the element is a <p> tag
        elif element.name == 'p':
            # Use the <p> tag as an answer
            current_answers.append(element.get_text())

            # Check if the <p> tag contains the text 'ví dụ'
            if 'ví dụ' in element.get_text().lower():
                next_element = element.find_next_sibling()
                if next_element:
                    # Check if the next element is a table
                    if next_element.name == 'table':
                        answer = extract_data_from_table(next_element)
                    else:
                        answer = next_element.get_text()

                    questions.append(element.get_text())
                    answers.append(answer)

    # Print the extracted questions and answers
    for question, answer in zip(questions, answers):
        print(f"Question: {question}\nAnswer: {answer}\n---")
else:
    print("Printing section not found.")
