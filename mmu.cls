%% ****************************************************
%% **                                                **
%% **        Manchester Metropolitan University      **
%% **                                                **
%% **    LaTeX Template for Thesis/Dissertation      **
%% **                                                **
%% **         Clinical RAG Chatbot Project           **
%% **                                                **
%% **     Adapted from University of Malta Template  **
%% **                                                **
%% ****************************************************

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mmu}[2025/08/06 v1.0.0 Manchester Metropolitan University, Dissertation/FYP/Thesis Template]


\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper,final]{memoir}


%% **************** Packages (Start) *********************

\RequirePackage[utf8]{inputenc}      % Required for inputting international characters
\RequirePackage[T1]{fontenc}         % Output font encoding for international characters
\RequirePackage{mathpazo}            % Use the Palatino font by default
\RequirePackage[english]{babel}      % Load babel if you're unsure about the default language - mostly to be safe
\RequirePackage{xcolor}              % For named colors
\RequirePackage{eso-pic}                 % required to place huge uni logo at the back - on title page
\RequirePackage[pdftex]{graphicx}    % For pictures
\RequirePackage{amssymb}             % At least, for black squares in bullet list
\RequirePackage[pdfusetitle]{hyperref}             % For hyperreferences
\RequirePackage[authoryear,semicolon,sort]{natbib} % for (Author, 2012; Smith, 2013) kind of references

\RequirePackage{longtable}           % For very long tables
\RequirePackage{pdflscape}			 % For landscape tables (instead of portrait)

\usepackage[defaultsans]{lato}		 % Sans font to use
\usepackage[printonlyused,withpage]{acronym}

%% ****************** Packages (End) *********************


%% ************ MMU Definitions (Start) *****************

\definecolor{OxfordBlue}{rgb}{0,0.106,0.329}   % Generic blue color
\definecolor{MMUBlue}{rgb}{0.118,0.086,0.337}  % MMU Primary Blue (#1E1656)
\definecolor{MMURed}{rgb}{0.827,0.196,0.325}   % MMU Secondary Red (if needed)

% Setup chosen Manchester Metropolitan University logo
\def\logo{{\includegraphics[width=35mm]{mmu_logo}}}

% The year and term the thesis is submitted 
\def\degreedate#1{\gdef\@degreedate{#1}}
% The full (unabbreviated) name of the degree
\def\degree#1{\gdef\@degree{#1}}
% The name of your supervisor
\def\supervisor#1{\gdef\@supervisor{#1}}
% The name of your co-supervisor
\def\cosupervisor#1{\gdef\@cosupervisor{#1}}
% The name of your department (e.g. Computer Science, AI, Data Science)
\def\department#1{\gdef\@department{#1}}
% The name of your faculty/school
\def\faculty#1{\gdef\@faculty{#1}}
% The subject code
\def\subjectcode#1{\gdef\@subjectcode{#1}}
% The tagline
\def\tagline#1{\gdef\@tagline{#1}}
% The document type, e.g. a dissertation or a thesis
\def\doctype#1{\gdef\@doctype{#1}}
\def\authorID#1{\gdef\@authorID{#1}}

%% ************* MMU Definitions (End) ******************


%% ************ Document Options (Start) *****************

\OnehalfSpacing                                  % One and a half line spacing

\setlength{\headsep}{1.5cm}                      % Add space between the header and text

\nouppercaseheads								 % Don't convert titles to Uppercase
\makepagestyle{mmupage}

\makeevenhead{mmupage}{\color{gray}\sffamily\small\leftmark}{}{\color{gray}\sffamily\small\rightmark}
\makeoddhead{mmupage}{\color{gray}\sffamily\small\leftmark}{}{\color{gray}\sffamily\small\rightmark}
\makeevenfoot{mmupage}{}{\thepage}{}  			 % MMU pagestyle, put page at bottom
\makeoddfoot{mmupage}{}{\thepage}{}
\makeheadfootruleprefix{mmupage}{\color{gray}}{}
\makeheadrule{mmupage}{\textwidth}{\normalrulethickness}
\makepsmarks{mmupage}{%
	\createmark{chapter}{left}{shownumber}{\@chapapp\ }{. \ }
	\createmark{section}{right}{shownumber}{}{. \ }
}


\setlrmarginsandblock{3.7cm}{2.5cm}{*}           % Set the page margins (for one and two sided docs) 
\checkandfixthelayout                            % Put layout into effect

\graphicspath{{./images/}}                       % Where to look for images (paths) ...
\DeclareGraphicsExtensions{.pdf,.jpeg,.png,.jpg} % Graphics extensions to load

\chapterstyle{ell} % how to draw the different chapter headings

\renewcommand*{\chapnumfont}{\sffamily\HUGE\bfseries\color{MMUBlue}}  % Chapter titles in MMU blue
\renewcommand*{\chaptitlefont}{\sffamily\HUGE\bfseries\color{MMUBlue}}
\setsecheadstyle{\sffamily\LARGE\bfseries\color{MMUBlue}}% Set \section style
\setsubsecheadstyle{\sffamily\Large\color{MMUBlue}}% Set \subsection style
\setsubsubsecheadstyle{\sffamily\large\color{MMUBlue}}% Set \subsection style
\setsecnumformat{\csname the#1\endcsname\enskip{\color{gray}\textbar}\enskip}

\newsubfloat{figure} % declares a new subfloat element which allows to use \subbottom commands

\renewcommand{\labelitemi}{\color{MMUBlue}\scriptsize$\blacksquare$}

\addto{\captionsenglish}{\renewcommand{\bibname}{References}} % because we are using babel we need this
                                                              % instead of just redefining bibname
\setlength{\bibitemsep}{\onelineskip}

\newcommand{\ra}[1]{\renewcommand{\arraystretch}{#1}} % to set array stretch within tables

\hypersetup{%
    colorlinks=false,
    pdfborder={0 0 0},    
}

\feetbelowfloat % we want the footnotes below floats, not wierdly above

\setsecnumdepth{subsubsection}  % three level depth - chapter, section, subsection, subsubsection
\settocdepth{subsection}

\renewcommand*{\cftappendixname}{Appendix\space}
\newcommand{\removelinebreaks}[1]{%
	\begingroup\def\\{ }#1\endgroup}

%% ************* Document Options (End) ******************


%% *************** Environments (Start) ******************

%% *** Title Page (Start) ***
% background image
\newcommand\AlCentroPagina[1]{%
\AddToShipoutPicture*{\AtPageCenter{%
\makebox(0,-50){\includegraphics[width=0.4\paperwidth]{#1}}}}}

% The front page
\renewcommand{\maketitle}
{\begingroup
\AlCentroPagina{mmu_logo_gray}
\parbox[b][0.95\textheight][t]{0.2\textwidth}{\raggedleft\logo} %% this 0.95 is important, otherwise page overflows in next page
\hspace*{2ex}
\textcolor{MMUBlue}{\rule{2pt}{0.95\textheight}}
\hspace*{2ex}
\parbox[b][0.95\textheight][t]{0.8\textwidth}{
    \setlength{\parindent}{0pt}
    \sffamily
    {\Huge\bfseries{\begin{Spacing}{1.15}\textcolor{MMUBlue}{\@title}\end{Spacing}}}
    \vspace*{2ex}
    {\large\textit{\@tagline}}\\[1in]
    
    {\LARGE\bfseries \textsc{\@author}} \\[1cm]
    {\large Supervised by \@supervisor} 
    \@ifundefined{@cosupervisor}{\\[2ex]}{\\[2ex]\large Co-supervised by \@cosupervisor}\\[1cm]
    {\large \@department} \\[1ex]
    {\large \@faculty} \\[1ex]
    {\large Manchester Metropolitan University} \\
    \vfill
    {\bfseries \@degreedate}\\[\baselineskip]
	\parbox[t]{0.80\linewidth}{\footnotesize\textit{A \@doctype\ submitted in partial fulfilment of the requirements for the degree of \@degree}.}    
}
\thispagestyle{empty}
\if@openright\cleardoublepage\else\clearpage\fi
\endgroup}
%% *** Title Page (End) ***

\newenvironment{acknowledgements}
{\renewcommand{\abstractname}{\Large\bfseries\sffamily\textcolor{MMUBlue}{Acknowledgements}}\abstract}
{\endabstract\if@openright\cleardoublepage\else\clearpage\fi}
       
\addto{\captionsenglish}{\renewcommand{\abstractname}{\Large\bfseries\sffamily\textcolor{MMUBlue}{Abstract}}}

\newenvironment{dedication}
  {\clearpage           % we want a new page
   \thispagestyle{empty}% no header and footer
   \vspace*{\stretch{1}}% some space at the top 
   \itshape             % the text is in italics
   \raggedleft          % flush to the right margin
   \textcolor{MMUBlue}
  }
  {\par % end the paragraph
   \vspace{\stretch{3}} % space at bottom is three times that at the top
   \if@openright\cleardoublepage\else\clearpage\fi
  }       

% MMU DECLARATION/ORIGINALITY STATEMENT
% Updated for Manchester Metropolitan University requirements

\newenvironment{originality}
{\clearpage\thispagestyle{empty}
\begin{figure}[h]
	\includegraphics[width=3in]{mmu_logo_full}
\end{figure}
\vspace{3mm}
{\noindent\Large\bfseries\sffamily\textcolor{MMUBlue}{Declaration}}\\ 

{\setlength{\parindent}{0em}
	
\textbf{Academic Integrity Declaration}\\

\noindent I hereby declare that this work is my own and has been written by me in its entirety. I have acknowledged all sources of information which have been used in the preparation of this report.

This report has not been submitted for assessment in any previous application for a degree or diploma or other qualification at this or any other university or institution.

I understand that if I am suspected of plagiarism or another form of academic misconduct, my work will be referred to the Academic Misconduct Panel for investigation.

I understand that Manchester Metropolitan University may retain copies of this work for educational and quality assurance purposes.}\\[5mm]

\bgroup
\def\arraystretch{1.2}%
\begin{tabular}{ll}
{\bfseries Faculty/School} & \@faculty \\
{\bfseries Degree} & \@degree \\
{\bfseries Title}  & \removelinebreaks{\@title} \\	
{\bfseries Student Name \@ifundefined{@authorID}{}{(Student ID)}} & \@author\ \@ifundefined{@authorID}{}{(\@authorID)} \\[0.7in]	
{\bfseries Student Signature} & \makebox[2in]{\hrulefill} \\
{\bfseries Date} & \today \\
\end{tabular}\\[1cm]
\egroup
\if@openright\cleardoublepage\else\clearpage\fi}


% COPYRIGHT
\newenvironment{copyrightenv}
{\clearpage
	~\vfill
	\thispagestyle{empty}
	
	\begin{figure}[h]
		\includegraphics[width=3in]{mmu_logo_full}
	\end{figure}
	
	\noindent Copyright \textcopyright\the\year\ Manchester Metropolitan University\\ % Copyright notice
	
	\noindent \textsc{www.mmu.ac.uk}\\ % URL
	
	\noindent \textit{First edition, \today} % Printing/edition date
	
	\if@openright\cleardoublepage\else\clearpage\fi}

%% **************** Environments (End) *******************
