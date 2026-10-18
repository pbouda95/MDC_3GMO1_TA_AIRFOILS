\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{base}

\LoadClass[a4paper,11pt]{book}

%extensions

%packages de base
%\RequirePackage[latin1]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[french]{babel}
\RequirePackage[top=2cm,left=2cm,right=2cm,bottom=2cm]{geometry}

\RequirePackage{fancyhdr}
\RequirePackage{mdframed}
\RequirePackage{pgf,tikz}
\RequirePackage{bigints}
\RequirePackage{hyperref}


%entete et pied de page
%\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\sffamily
\numberwithin{equation}{section}
