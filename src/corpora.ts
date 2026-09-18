/**
 * corpora.ts — the texts the demo trains on out of the box.
 *
 * The default is deliberately small and highly structured: a list of given
 * names gives a tiny model enough regularity to invent new ones within seconds,
 * which is the honest version of "watch it learn". A larger, messier text needs
 * a bigger model and far more time, and a browser cannot give it that — the page
 * says so rather than pretending otherwise.
 *
 * The dialogue set was written for this demo. The names are a hand-written list
 * of common given names (a list of names is a fact, not a work).
 */

export interface Corpus {
  key: string;
  label: string;
  note: string;
  text: string;
}

export const NAME_LIST = `Aaron
Abigail
Adam
Adrian
Aiden
Alan
Albert
Alexa
Alexander
Alexis
Alice
Alicia
Alison
Allan
Allen
Amanda
Amber
Amy
Andrea
Andrew
Angela
Anita
Ann
Anna
Anne
Annette
Anthony
April
Archie
Arthur
Ashley
Audrey
Austin
Barbara
Barry
Beatrice
Benjamin
Bernard
Beth
Beverly
Bill
Blake
Bonnie
Bradley
Brandon
Brenda
Brendan
Brian
Bridget
Bruce
Bryan
Caleb
Cameron
Camila
Carl
Carla
Carlos
Carmen
Carol
Caroline
Carolyn
Carrie
Casey
Catherine
Cathy
Cecil
Cedric
Chad
Charlene
Charles
Charlotte
Chase
Cheryl
Chester
Chloe
Chris
Christian
Christina
Christine
Christopher
Cindy
Claire
Clara
Clarence
Claude
Claudia
Clayton
Clifford
Clint
Cody
Cole
Colin
Colleen
Connor
Connie
Conrad
Constance
Cooper
Cora
Corey
Courtney
Craig
Crystal
Curtis
Cynthia
Daisy
Dale
Dallas
Damon
Dan
Dana
Daniel
Danielle
Danny
Darlene
Darrell
Darren
Daryl
Dave
David
Dawn
Dean
Deborah
Debra
Declan
Delia
Denis
Dennis
Derek
Derrick
Desmond
Diana
Diane
Dominic
Don
Donald
Donna
Doreen
Doris
Dorothy
Douglas
Duane
Dustin
Dylan
Earl
Edgar
Edith
Edmund
Edward
Edwin
Eileen
Elaine
Eleanor
Elena
Eli
Elias
Elijah
Elinor
Elizabeth
Ella
Ellen
Elliot
Elmer
Eloise
Elsie
Emil
Emily
Emma
Emmett
Enid
Eric
Erica
Erin
Ernest
Esther
Ethan
Ethel
Eugene
Eunice
Eva
Evan
Evelyn
Everett
Faith
Felix
Fiona
Florence
Floyd
Frances
Francis
Frank
Franklin
Fred
Freda
Frederick
Gabriel
Gabrielle
Gail
Garrett
Gary
Gavin
Gene
Geoffrey
George
Gerald
Geraldine
Gilbert
Gina
Gladys
Glen
Glenn
Gloria
Gordon
Grace
Graham
Grant
Greg
Gregory
Gretchen
Guy
Gwen
Hannah
Harold
Harriet
Harry
Harvey
Hazel
Heather
Hector
Helen
Henrietta
Henry
Herbert
Herman
Hilda
Holly
Horace
Howard
Hugh
Hugo
Ian
Ibrahim
Ida
Igor
Irene
Iris
Irma
Irving
Isaac
Isabel
Isabella
Ivan
Ivy
Jack
Jackie
Jacob
Jacqueline
Jade
Jaime
Jake
James
Jamie
Jan
Jane
Janet
Janice
Jared
Jasmine
Jason
Jasper
Javier
Jay
Jean
Jeanette
Jeff
Jeffrey
Jenna
Jennifer
Jenny
Jeremy
Jerome
Jerry
Jesse
Jessica
Jessie
Jill
Jim
Jimmy
Joan
Joanna
Joanne
Joe
Joel
Joey
Johanna
John
Jonah
Jonathan
Jordan
Jose
Joseph
Josephine
Joshua
Joyce
Juan
Judith
Judy
Julia
Julian
Julie
Julius
June
Justin
Kaitlyn
Karen
Katherine
Kathleen
Kathryn
Kathy
Katie
Kayla
Keith
Kelly
Ken
Kendra
Kenneth
Kent
Kerry
Kevin
Kieran
Kim
Kimberly
Kirk
Kristen
Kristin
Kyle
Lance
Larry
Laura
Lauren
Laurie
Lawrence
Leah
Lee
Leo
Leon
Leonard
Leroy
Leslie
Lester
Levi
Lewis
Liam
Lila
Lillian
Lily
Linda
Lindsay
Lionel
Lisa
Lloyd
Logan
Lois
Lola
Lorraine
Louis
Louise
Lucas
Lucia
Lucy
Luke
Luther
Lydia
Lyle
Lynda
Lynn
Mabel
Mackenzie
Maddie
Madeline
Mae
Maggie
Malcolm
Mandy
Manuel
Marc
Marcia
Marcus
Margaret
Maria
Marian
Marie
Marilyn
Marion
Marjorie
Mark
Marlene
Marsha
Marshall
Martin
Marty
Marvin
Mary
Mason
Matthew
Maureen
Maurice
Max
Maxine
Megan
Melanie
Melinda
Melissa
Melvin
Mercedes
Meredith
Mia
Micah
Michael
Michele
Michelle
Mickey
Miguel
Mildred
Miles
Millicent
Milton
Mindy
Miriam
Mitchell
Molly
Monica
Morgan
Morris
Moses
Muriel
Murray
Myra
Myrtle
Nadia
Nancy
Naomi
Natalie
Nathan
Nathaniel
Neal
Neil
Nelson
Nicholas
Nicole
Nina
Noah
Noel
Nora
Norma
Norman
Norton
Olga
Olive
Oliver
Olivia
Omar
Opal
Oscar
Owen
Pablo
Pamela
Patricia
Patrick
Patsy
Paul
Paula
Paulette
Pearl
Pedro
Peggy
Penny
Percy
Perry
Peter
Phil
Philip
Phillip
Phyllis
Pierre
Polly
Preston
Priscilla
Quentin
Rachel
Rafael
Ralph
Ramona
Randall
Randolph
Randy
Raquel
Ray
Raymond
Rebecca
Regina
Reginald
Rene
Rhonda
Ricardo
Richard
Rick
Ricky
Rita
Robert
Roberta
Roberto
Robin
Rodney
Roger
Roland
Ronald
Ronnie
Rosa
Rosalie
Rose
Rosemary
Ross
Roy
Ruben
Ruby
Rudolph
Rufus
Russell
Ruth
Ryan
Sabrina
Sadie
Sally
Salvador
Sam
Samantha
Samuel
Sandra
Sandy
Santiago
Sarah
Saul
Scott
Sean
Selena
Selma
Serena
Seth
Shane
Shannon
Sharon
Shaun
Shawn
Sheila
Shelby
Shelley
Sherman
Sherry
Shirley
Sidney
Simon
Sonia
Sonja
Sophia
Sophie
Spencer
Stacey
Stacy
Stanley
Stella
Stephanie
Stephen
Sterling
Steve
Steven
Stewart
Stuart
Sue
Susan
Susanna
Susie
Suzanne
Sylvia
Sylvie
Tabitha
Tamara
Tanya
Tara
Ted
Teresa
Terrence
Terry
Tess
Thelma
Theodore
Theresa
Thomas
Tiffany
Timothy
Tina
Toby
Todd
Tom
Tomas
Tommy
Tony
Tracy
Travis
Trevor
Tricia
Trisha
Troy
Tyler
Ursula
Valerie
Vanessa
Vera
Vernon
Veronica
Victor
Victoria
Vincent
Viola
Violet
Virginia
Vivian
Vivien
Wade
Wallace
Walter
Wanda
Warren
Wayne
Wendell
Wendy
Wesley
Whitney
Wilbur
Wilfred
Willard
William
Willie
Wilma
Winifred
Winston
Wyatt
Xavier
Yolanda
Yvette
Yvonne
Zachary
Zoe`;

export const DIALOGUE = `Ana: are you coming tonight
Ben: i can't, i'm working late
Ana: again?
Ben: it's the second week of the month, that's when it always happens
Ana: right. i forgot
Ben: i'll make it up at the weekend
Ana: you said that last weekend
Ben: i know. this time i mean it
Ana: i've heard that one before
Ben: fair
Ana: are you eating properly
Ben: mostly
Ana: what does mostly mean
Ben: it means i had a sandwich at my desk
Ana: that's not a meal
Ben: it had lettuce in it
Ana: lettuce is not a meal either
Ben: fine, i'll cook something when i get home
Ana: it'll be midnight
Ben: midnight pasta is a tradition
Ana: it's a bad tradition
Ben: it's a tradition
Ana: did you call your mother
Ben: no
Ana: you said you would
Ben: i will
Ana: when
Ben: soon
Ana: that's not a day
Ben: i'll call her on sunday
Ana: she'll ask about me
Ben: she always asks about you
Ana: what do you tell her
Ben: the truth
Ana: which is
Ben: that you're the reason i'm ever on time anywhere
Ana: that's a lie
Ben: it's a kind lie
Ana: how was the meeting
Ben: long
Ana: useful
Ben: mixed
Ana: what does mixed mean
Ben: it means they liked the work and changed the deadline
Ana: that sounds normal
Ben: it is. that's the problem
Ana: are you free on thursday
Ben: i could be
Ana: could be is not free
Ben: i'm free after six
Ana: i'll take it
Ben: are you bringing anyone
Ana: just me
Ben: good
Ana: is that good
Ben: it's easier to talk
Ana: we always talk
Ben: we talk about work
Ana: we could talk about something else
Ben: like what
Ana: like anything
Ben: give me a topic
Ana: the weather
Ben: it's raining
Ana: see, that was easy
Ben: that was one sentence
Ana: it's a start
Ben: i'll see you thursday
Ana: don't be late
Ben: i'm never late
Ana: you're always late
Ben: i'm never late on thursdays
Ana: we'll see
Ben: we will
Ana: how did it go
Ben: better than i expected
Ana: that's not saying much
Ben: it's saying something
Ana: what did they say
Ben: they want to do it again next month
Ana: that's good
Ben: it's work
Ana: work is good
Ben: work is money
Ana: money is good too
Ben: i never said it wasn't
Ana: you look tired
Ben: i am tired
Ana: go to bed
Ben: i will in a minute
Ana: you always say that
Ben: and i always mean it
Ana: do you want tea
Ben: please
Ana: one sugar
Ben: two
Ana: you're going to rot
Ben: probably
Ana: here
Ben: thank you
Ana: you're welcome
Ben: what time is it
Ana: late
Ben: how late
Ana: past the point of asking
Ben: i should go
Ana: you should
Ben: i'll text you when i get in
Ana: you won't
Ben: i will
Ana: you never do
Ben: this time i will
Ana: goodnight
Ben: goodnight`;

export const CORPORA: Record<string, Corpus> = {
  names: {
    key: 'names',
    label: 'Given names',
    note: 'about six hundred names, one per line. The model learns to invent new ones.',
    text: NAME_LIST,
  },
  dialogue: {
    key: 'dialogue',
    label: 'Dialogue',
    note: 'a short two-person conversation, written for this demo. The model learns the shape of a turn.',
    text: DIALOGUE,
  },
};

export const DEFAULT_CORPUS = 'names';
