import os


def create_navigation_html(root_dir, save_dir):
    sections = {}
    level = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            thing = os.path.basename(os.path.dirname(dirpath))
            if not thing.endswith(".io") and thing != "gram_eigens":
                sections.setdefault(os.path.basename(os.path.dirname(dirpath)), {})
        for filename in filenames:
            if filename.endswith(".html"):
                sections.setdefault(os.path.basename(os.path.dirname(dirpath)), {})[os.path.basename(dirpath)] = os.path.join(dirpath, filename)
        level -= 1
    html = """
<!DOCTYPE html>
<html>
  <head>
    <title>Modularity Experiments</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
  </head>
  <body>
    <header>
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
"""
    for section in sections:
        html += f"            <a class='nav-item nav-link' href='#{section}'>{section}</a>\n"

    html += """        </nav>
    </header>
     <main>
        <h1>Welcome to My HTML Pages</h1>
        <div id="accordion">
"""
    for section, pages in sections.items():
        html += f"            <div class='card'>\n"
        html += f"                <div class='card-header' id='heading{section}'>\n"
        html += f"                    <h5 class='mb-0'>\n"
        html += f"                        <button class='btn btn-link' data-toggle='collapse' data-target='#collapse{section}' aria-expanded='true' aria-controls='collapse{section}'>\n"
        html += f"                            {section}\n"
        html += f"                        </button>\n"
        html += f"                    </h5>\n"
        html += f"                </div>\n"
        html += f"                <div id='collapse{section}' class='collapse' aria-labelledby='heading{section}' data-parent='#accordion'>\n"
        html += "                    <ul class='list-group list-group-flush'>\n"
        for page in pages.items():
            url = page[1].replace("C:/Users/avery/Projects/github.io/", "https://")
            html += f"                        <li class='list-group-item'><a href='{url}'>{page[0]}</a></li>\n"
        html += "                    </ul>\n"
        html += "                </div>\n"
        html += "            </div>\n"
    html += """        </div>
    </main>
  </body>
</html>
"""

    with open(f"{save_dir}/index.html", "w") as file:
        file.write(html)


if __name__ == "__main__":
    root_dir = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io\gram_eigens".replace("\\", "/")
    save_dir = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io".replace("\\", "/")
    create_navigation_html(root_dir, save_dir)


