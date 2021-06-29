\LoadClass[12pt]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[spanish,es-tabla]{babel}
\RequirePackage{titlesec}
\RequirePackage{titling}
\RequirePackage{graphicx}
\RequirePackage{indentfirst}
\RequirePackage{float}
\RequirePackage{setspace} 
\RequirePackage{afterpage}
\RequirePackage{tabularx}
\RequirePackage{fancyhdr}
\RequirePackage[labelfont=bf,textfont=bf]{caption}
\RequirePackage[toc,page]{appendix}

\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{4}

%\pagestyle{fancy}
%\rfoot{\thepage}

% paginas en blanco
\newcommand\blankpage{%
    \null
    \thispagestyle{empty}%
    \addtocounter{page}{-1}%
    \newpage}

% margenes
\RequirePackage[a4paper, left = 1.18in, right = 0.79in, top = 1.18in, bottom = 0.79in]{geometry}

\newcommand{\MONTH}{%
  \ifcase\the\month
  \or Enero % 1
  \or Febrero % 2
  \or Marzo % 3
  \or Abril % 4
  \or Mayo % 5
  \or Junio % 6
  \or Julio % 7
  \or Agosto % 8
  \or Septiembre % 9
  \or Octubre % 10
  \or Noviembre % 11
  \or Diciembre % 12
  \fi}

%parrafo%
\parindent=0.49in
\parskip=6pt
\onehalfspacing

%sections%
\renewcommand\section{\@startsection{section}{1}{\z@}%
                       {-18\p@ \@plus -4\p@ \@minus -4\p@}%
                       {12\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\newpage\normalfont\large\bfseries\boldmath
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                       {-18\p@ \@plus -4\p@ \@minus -4\p@}%
                       {8\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\normalfont\normalsize\bfseries\boldmath
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                       {-18\p@ \@plus -4\p@ \@minus -4\p@}%
                       {8\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\normalfont\normalsize\bfseries\boldmath
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\titleformat{\paragraph}
{\normalfont\normalsize\bfseries}{\theparagraph}{1em}{}
\titlespacing*{\paragraph}
{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}


  
