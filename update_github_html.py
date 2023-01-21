import os


# def create_navigation_html(root_dir):
#     sections = {}
#     level = 0
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         level += len(dirnames)
#         for filename in filenames:
#             if filename.endswith(".html"):
#                 sections.setdefault(os.path.basename(os.path.dirname(dirpath)), {})[os.path.basename(dirpath)] = os.path.join(dirpath, filename)
#         level -= 1
#
#     html = """
# <!DOCTYPE html>
# <html>
#   <head>
#     <title>My HTML Pages</title>
#     <link rel="stylesheet" type="text/css" href="styles.css">
#   </head>
#   <body>
#     <header>
#         <nav>
# """
#     for section in sections:
#         html += f"            <a href='#{section}'>{section}</a>\n"
#
#     html += """        </nav>
#     </header>
#     <main>
#         <h1>Welcome to My HTML Pages</h1>
# """
#     for section, pages in sections.items():
#         html += f"        <section class='level-{level}' id='{section}'>\n"
#         html += f"            <h2>{section}</h2>\n"
#         html += "            <ul>\n"
#         for page in pages.items():
#             html += f"                <li><a href='{page[1]}'>{page[0]}</a></li>\n"
#         html += "            </ul>\n"
#         html += "        </section>\n"
#         level+=1
#
#     html += """    </main>
#   </body>
# </html>
# """
#
#     with open("index.html", "w") as file:
#         file.write(html)


def create_navigation_html3(root_dir, save_dir):
    sections = {}
    level = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            sections.setdefault(os.path.basename(os.path.dirname(dirpath)), {})
        for filename in filenames:
            if filename.endswith(".html"):
                sections.setdefault(os.path.basename(os.path.dirname(dirpath)), {})[os.path.basename(dirpath)] = os.path.join(dirpath, filename)
        level -= 1
    html = """
<!DOCTYPE html>
<html>
  <head>
    <title>My HTML Pages</title>
    <link rel="stylesheet" type="text/css" href="styles.css">
  </head>
  <body>
    <header>
        <nav>
"""
    for section in sections:
        html += f"            <a href='#{section}'>{section}</a>\n"

    html += """        </nav>
    </header>
    <main>
        <h1>Welcome to My HTML Pages</h1>
"""
    for section, pages in sections.items():
        html += f"        <section class='level-{level}' id='{section}'>\n"
        html += f"            <h2>{section}</h2>\n"
        html += "            <ul>\n"
        for page in pages.items():
            url = page[1].replace("C:/Users/avery/Projects/github.io/", "https://")
            html += f"                <li><a href='{url}'>{page[0]}</a></li>\n"
        html += "            </ul>\n"
        html += "        </section>\n"
        level += 1

    html += """    </main>
  </body>
</html>
"""

    with open(f"{save_dir}/index.html", "w") as file:
        file.write(html)


if __name__ == "__main__":
    root_dir = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io\gram_eigens".replace("\\", "/")
    save_dir = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io".replace("\\", "/")
    create_navigation_html3(root_dir, save_dir)
# def create_navigation_html2(root_dir):
#     sections = {}
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for dirname in dirnames:
#             sections.setdefault(dirname, {})
#         for filename in filenames:
#             if filename.endswith(".html"):
#                 sections[os.path.basename(dirpath)][filename] = os.path.join(dirpath)
#
#     html = """
# <!DOCTYPE html>
# <html>
#   <head>
#     <title>My HTML Pages</title>
#     <link rel="stylesheet" type="text/css" href="styles.css">
#   </head>
#   <body>
#     <header>
#         <nav>
# """
#     for section in sections:
#         html += f"            <a href='#{section}'>{section}</a>\n"
#
#     html += """        </nav>
#     </header>
#     <main>
#         <h1>Welcome to My HTML Pages</h1>
# """
#     for section, pages in sections.items():
#         html += f"        <section id='{section}'>\n"
#         html += f"            <h2>{section}</h2>\n"
#         html += "            <ul>\n"
#         for page in pages.items():
#             html += f"                <li><a href='{page[1]}'>{page[0]}</a></li>\n"
#         html += "            </ul>\n"
#         html += "        </section>\n"
#
#     html += """    </main>
#   </body>
# </html>
# """
#
#     with open("index.html", "w") as file:
#         file.write(html)

#
# def create_navigation_html1(root_dir):
#     sections = {}
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith(".html"):
#                 sections.setdefault(os.path.basename(dirpath), []).append(os.path.join(dirpath))
#
#     html = """
# <!DOCTYPE html>
# <html>
#   <head>
#     <title>My HTML Pages</title>
#     <link rel="stylesheet" type="text/css" href="styles.css">
#   </head>
#   <body>
#     <header>
#         <nav>
# """
#     for section in sections:
#         html += f"            <a href='#{section}'>{section}</a>\n"
#
#     html += """        </nav>
#     </header>
#     <main>
#         <h1>Welcome to My HTML Pages</h1>
# """
#     for section, pages in sections.items():
#         html += f"        <section id='{section}'>\n"
#         html += f"            <h2>{section}</h2>\n"
#         html += "            <ul>\n"
#         for page in pages:
#             page = page.replace("C:/Users/avery/Projects/github.io/", "https://")
#             html += f"                <li><a href='{page}'>{os.path.basename(page)}</a></li>\n"
#         html += "            </ul>\n"
#         html += "        </section>\n"
#
#     html += """    </main>
#   </body>
# </html>
# """
#
#     with open("index.html", "w") as file:
#         file.write(html)


if __name__ == "__main__":
    root_dir = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io\gram_eigens".replace("\\", "/")
    save_dir = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io".replace("\\", "/")
    create_navigation_html3(root_dir, save_dir)


